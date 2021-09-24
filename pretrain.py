#!/usr/bin/env python
from __future__ import print_function
import os
import time
import yaml
import pprint
import random
import pickle
import shutil
import inspect
from collections import OrderedDict, defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.distributions.uniform import Uniform
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR

import apex
import wandb

from model.port import MORT
from utils import count_params, import_class, init_seed, random_rot_mat, \
    get_masked_input_and_labels, get_motion, repeat_rot_mat
from loss import sym_reg, constraint_reg, ReconLoss, feature_transform_reguliarzer, CosineSimilarity
from opts import get_pretrain_parser
from vis import plot_attention_weights


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            # Added control through the command line
            logdir = os.path.join(arg.work_dir, 'trainlogs')
            if not arg.debug:
                # logdir = arg.model_saved_name
                if os.path.isdir(logdir):
                    print(f'log_dir {logdir} already exists')
                    if arg.assume_yes:
                        answer = 'y'
                    else:
                        answer = input('delete it? [y]/n:')
                    if answer.lower() in ('y', ''):
                        shutil.rmtree(logdir)
                        print('Dir removed:', logdir)
                    else:
                        print('Dir not removed:', logdir)

                self.train_writer = SummaryWriter(os.path.join(logdir, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(logdir, 'val'), 'val')
            else:
                self.train_writer = SummaryWriter(os.path.join(logdir, 'debug'), 'debug')

        self.num_motions = len(self.arg.motion_time_steps) + 1
        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_data()
        self.load_lr_scheduler()

        self.global_step = 0
        self.lr = self.arg.base_lr
        self.min_loss = np.inf

        if self.arg.rot:
            self.random_dist = Uniform(low=-self.arg.max_rot, high=self.arg.max_rot)

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.print_log(f'{len(self.arg.device)} GPUs available, using DataParallel')
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device
                )

    def load_model(self):
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device

        # Copy model file and main
        shutil.copy2(inspect.getfile(MORT), self.arg.work_dir)
        shutil.copy2(os.path.join('.', __file__), self.arg.work_dir)

        self.model = MORT(
            channels=3*self.num_motions,
            dim=self.arg.embd_dim,
            mlp_dim=self.arg.embd_dim*2,
            depth=self.arg.n_layers,
            heads=self.arg.n_heads,
            dropout=0.1,
            emb_dropout=0.1
        ).cuda(output_device)
        if self.arg.loss == 'huber':
            self.loss = nn.SmoothL1Loss().cuda(output_device)
        elif self.arg.loss == 'l1':
            self.loss = nn.L1Loss().cuda(output_device)
        elif self.arg.loss == 'mse':
            self.loss = nn.MSELoss().cuda(output_device)
        elif self.arg.loss == 'l2':
            self.loss = ReconLoss().cuda(output_device)
        elif self.arg.loss == 'cosin':
            self.loss = CosineSimilarity().cuda(output_device)
        self.print_log(f'Model total number of params: {count_params(self.model)}')

        if self.arg.weights:
            try:
                self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            except:
                print('Cannot parse global_step from model weights filename')
                self.global_step = 0

            self.print_log(f'Loading weights from {self.arg.weights}')
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log(f'Sucessfully Remove Weights: {w}')
                else:
                    self.print_log(f'Can Not Remove Weights: {w}')

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {self.arg.optimizer}')

        # Load optimizer states if any
        if self.arg.checkpoint is not None:
            self.print_log(f'Loading optimizer states from: {self.arg.checkpoint}')
            self.optimizer.load_state_dict(torch.load(self.arg.checkpoint)['optimizer_states'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.print_log(f'Starting LR: {current_lr}')
            self.print_log(f'Starting WD1: {self.optimizer.param_groups[0]["weight_decay"]}')
            if len(self.optimizer.param_groups) >= 2:
                self.print_log(f'Starting WD2: {self.optimizer.param_groups[1]["weight_decay"]}')

    def load_lr_scheduler(self):
        if self.arg.lr_scheduler == 'cosin':
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.arg.num_warmup_steps,
                num_training_steps=len(self.data_loader['train'])* self.arg.num_epoch
            )
        elif self.arg.lr_scheduler == 'exp':
            self.lr_scheduler = ExponentialLR(self.optimizer, gamma=self.arg.lr_decay)
        if self.arg.checkpoint is not None:
            scheduler_states = torch.load(self.arg.checkpoint)['lr_scheduler_states']
            self.print_log(f'Loading LR scheduler states from: {self.arg.checkpoint}')
            self.lr_scheduler.load_state_dict(scheduler_states)
            self.print_log(f'Starting last epoch: {scheduler_states["last_epoch"]}')
            self.print_log(f'Loaded milestones: {scheduler_states["last_epoch"]}')

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        def worker_seed_fn(worker_id):
            # give workers different seeds
            return init_seed(self.arg.seed + worker_id + 1)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    f'{self.arg.data_root}/{self.arg.train_dataset}/{self.arg.train_data_case}/train_data_{self.arg.train_data_type}.npy',
                    f'{self.arg.data_root}/{self.arg.train_dataset}/{self.arg.train_data_case}/train_label.pkl',
                    downsample=self.arg.train_ds_downsample,
                    motion_time_steps=self.arg.motion_time_steps,
                    jittering=self.arg.jittering,
                    window_size=64,
                    p_interval=[0.5, 1],
                    crop_window=self.arg.crop_window,
                    vectorize=self.arg.vectorize,
                    localize=self.arg.localize
                ),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                f'{self.arg.data_root}/{self.arg.test_dataset}/{self.arg.test_data_case}/val_data_{self.arg.test_data_type}.npy',
                f'{self.arg.data_root}/{self.arg.test_dataset}/{self.arg.test_data_case}/val_label.pkl',
                downsample=self.arg.test_ds_downsample,
                motion_time_steps=self.arg.motion_time_steps,
                window_size=64,
                p_interval=[0.95],
                crop_window=self.arg.crop_window,
                vectorize=self.arg.vectorize,
                localize=self.arg.localize
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, epoch, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)
        return out_path

    def save_checkpoint(self, epoch, out_folder='checkpoints'):
        state_dict = {
            'epoch': epoch,
            'optimizer_states': self.optimizer.state_dict(),
            'lr_scheduler_states': self.lr_scheduler.state_dict(),
        }

        return self.save_states(epoch, state_dict, out_folder, 'recent.pt')

    def save_weights(self, epoch, out_folder='weights'):
        state_dict = self.model.state_dict()
        weights = OrderedDict([
            [k.split('module.')[-1], v.cpu()]
            for k, v in state_dict.items()
        ])

        return self.save_states(epoch, weights, out_folder, 'recent.pt')

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_values = []
        self.train_writer.add_scalar('epoch', epoch + 1, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        current_lr = self.optimizer.param_groups[0]['lr']
        self.print_log(f'Training epoch: {epoch + 1}, LR: {current_lr:.6f}')

        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            # real_batch_size = self.arg.forward_batch_size
            real_batch_size = self.arg.batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):
                left = i * real_batch_size
                right = left + real_batch_size
                batch_data, _ = data[left:right], label[left:right]

                N, C, T, V, M, A = batch_data.size()
                batch_data = batch_data.permute(0, 4, 2, 3, 5, 1).contiguous().view(N*M*T, V, A*C)

                if self.arg.nonzero_mask:
                    nonzero_mask = batch_data.view(N*M*T, -1).count_nonzero(dim=-1) !=0
                    batch_data = batch_data[nonzero_mask]

                if self.arg.rot:
                    rot_mat = random_rot_mat(batch_data.shape[0], self.random_dist).to(batch_data.device) # (NMT, AC, AC)
                    rot_mat = repeat_rot_mat(rot_mat, self.num_motions)
                    batch_data = batch_data.transpose(1, 2) # (NMT, AC, V)
                    batch_data = torch.bmm(rot_mat, batch_data) # (NMT, AC, V)
                    batch_data = batch_data.transpose(1, 2) #(NMT, V, AC)

                # shuffle
                rand_ids = torch.randperm(batch_data.shape[0])
                batch_data = batch_data[rand_ids]

                batch_data, gt = get_masked_input_and_labels(
                    batch_data,
                    mask_value=self.arg.mask_value,
                    mask_p=self.arg.mask_p,
                    mask_remain_p=self.arg.mask_remain_p,
                    mask_random_p=self.arg.mask_random_p,
                )

                # forward
                loss = 0
                output, _, _ = self.model(batch_data)

                pose_loss = self.loss(output[:,:,:3], gt[:,:,:3]) / splits
                loss += pose_loss
                self.train_writer.add_scalar('pose_loss', pose_loss.item() * splits, self.global_step)
                for i, t in enumerate(self.arg.motion_time_steps):
                    motion_loss = self.loss(output[:,:,3*(i+1):3*(i+2)], gt[:,:,3*(i+1):3*(i+2)]) / splits
                    loss += self.arg.lambda_motion[i] * motion_loss
                    self.train_writer.add_scalar(f'motion_time_{t}_loss', motion_loss.item() * splits, self.global_step)

                if self.arg.half:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())
                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.6f}')

                self.train_writer.add_scalar('total_loss', loss.item() * splits, self.global_step)

            #####################################

            nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()
            if self.arg.lr_scheduler == 'cosin':
                self.lr_scheduler.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        if self.arg.lr_scheduler != 'cosin':
            self.lr_scheduler.step()

        # statistics of time consumption and loss

        mean_loss = np.mean(loss_values)
        num_splits = self.arg.batch_size // self.arg.forward_batch_size
        self.print_log(f'\tMean training loss: {mean_loss:.6f} (BS {self.arg.batch_size}: {mean_loss * num_splits:.6f}).')


    def eval(self, epoch, loader_name=['test'], wrong_file=None, result_file=None):
        # Skip evaluation if too early
        if epoch + 1 < self.arg.eval_start:
            return

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')

        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval epoch: {epoch + 1}')
            for ln in loader_name:
                sym_loss_values = []
                const_loss_values = []
                pose_loss_values = []
                vel_loss_values = []
                acc_loss_values = []
                motion_loss_values = {t:[] for t in self.arg.motion_time_steps}
                step = 0
                process = tqdm(self.data_loader[ln], dynamic_ncols=True)
                for batch_idx, (data, label, index) in enumerate(process):
                    data = data.float().cuda(self.output_device)
                    N, C, T, V, M, A = data.size()
                    data = data.permute(0, 4, 2, 3, 5, 1).contiguous().view(N*M*T, V, A*C)

                    if self.arg.nonzero_mask:
                        nonzero_mask = data.view(N*M*T, -1).count_nonzero(dim=-1) !=0
                        data = data[nonzero_mask]

                    data, gt = get_masked_input_and_labels(
                        data,
                        mask_value=self.arg.mask_value,
                        mask_p=self.arg.mask_p,
                        mask_remain_p=self.arg.mask_remain_p,
                        mask_random_p=self.arg.mask_random_p,
                    )

                    output, attns, _ = self.model(data)

                    if batch_idx == 0:
                        plot_attention_weights([attn[0].cpu() for attn in attns])
                        wandb.log({"Attentions": plt}, step=self.global_step)

                    pose_loss = self.loss(output[:,:,:3], gt[:,:,:3])
                    pose_loss_values.append(pose_loss.item())
                    for i, t in enumerate(self.arg.motion_time_steps):
                        motion_loss = self.loss(output[:,:, 3*(i+1):3*(i+2)], gt[:,:,3*(i+1):3*(i+2)])
                        motion_loss_values[t].append(motion_loss.item())

                    sym_loss = sym_reg(output[:,:,:3])
                    const_loss = constraint_reg(output[:,:,:3])
                    sym_loss_values.append(sym_loss.item())
                    const_loss_values.append(const_loss.item())


            sym_loss = np.mean(sym_loss_values)
            const_loss = np.mean(const_loss_values)
            pose_loss = np.mean(pose_loss_values)
            loss = pose_loss
            for i, t in enumerate(self.arg.motion_time_steps):
                motion_loss = np.mean(motion_loss_values[t])
                loss += self.arg.lambda_motion[i] * motion_loss
                self.val_writer.add_scalar(f'motion_time_{t}_loss', motion_loss, self.global_step)

            self.val_writer.add_scalar('total_loss', loss, self.global_step)
            self.val_writer.add_scalar('sym_loss', sym_loss, self.global_step)
            self.val_writer.add_scalar('const_loss', const_loss, self.global_step)
            self.val_writer.add_scalar('pose_loss', pose_loss, self.global_step)

            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {loss}.')

            weights_path = self.save_weights(epoch + 1)
            ckpt_path = self.save_checkpoint(epoch + 1)
            if epoch> 20 and loss < self.min_loss:
                self.min_loss = loss
                weights_name = f'weights-{epoch}-{int(self.global_step)}.pt'
                checkpoint_name = f'checkpoint-{epoch}-fwbz{self.arg.forward_batch_size}-{int(self.global_step)}.pt'
                weights_dir = os.path.dirname(weights_path)
                ckpt_dir = os.path.dirname(ckpt_path)
                shutil.copy(weights_path, f'{weights_dir}/{weights_name}')
                shutil.copy(ckpt_path, f'{ckpt_dir}/{checkpoint_name}')


        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):
        if self.arg.phase == 'train':
            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                if epoch % self.arg.eval_interval == 0:
                    self.eval(epoch, loader_name=['test'])

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

        elif self.arg.phase == 'test':
            if not self.arg.debug:
                wf = os.path.join(self.arg.work_dir, 'wrong-samples.txt')
                rf = os.path.join(self.arg.work_dir, 'right-samples.txt')
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')

            self.print_log(f'Model:   {self.arg.model}')
            self.print_log(f'Weights: {self.arg.weights}')

            self.eval(
                epoch=0,
                loader_name=['test'],
                wrong_file=wf,
                result_file=rf
            )

            self.print_log('Done.\n')


def main():
    # parser arguments
    parser = get_pretrain_parser()
    arg = parser.parse_args()
    # initialize wandb
    wandb.init(
        project="stport_pretrain_new",
        entity="chibros",
        sync_tensorboard=True,
        dir=arg.log_dir
    )
    arg.work_dir = wandb.run.dir
    wandb.config.update(arg)
    init_seed(arg.seed)
    # execute process
    processor = Processor(arg)
    processor.start()

if __name__ == '__main__':
    main()
