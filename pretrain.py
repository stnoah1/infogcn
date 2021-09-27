#!/usr/bin/env python
from __future__ import print_function

import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from opts import get_pretrain_parser
from loss import ReconLoss
from utils import get_masked_input_and_labels
from model.port import MORT
from vis import plot_attention_weights

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import apex
import wandb

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class Processor():
    """
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.debug:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_loss = np.inf
        self.best_loss_epoch = 0

        self.model = self.model.cuda(self.output_device)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.arg.num_warmup_steps,
            num_training_steps=len(self.data_loader['train'])* self.arg.num_epoch
        )


        if self.arg.half:
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{self.arg.amp_opt_level}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_path = f'data/{self.arg.dataset}/NTU60_{self.arg.datacase}_aligned.npz'
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(
                    data_path=data_path,
                    split='train',
                    window_size=64,
                    p_interval=[0.5, 1],
                    random_rot=True
                ),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                window_size=64,
                p_interval=[0.95],
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device

        self.model = MORT(
            channels=3,
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

        if self.arg.weights:
            self.global_step = int(self.arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            N, C, T, V, M = data.shape
            data = data.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)

            # shuffle
            rand_ids = torch.randperm(data.shape[0])
            data = data[rand_ids]

            data, gt = get_masked_input_and_labels(
                data,
                mask_value=self.arg.mask_value,
                mask_p=self.arg.mask_p,
                mask_remain_p=self.arg.mask_remain_p,
                mask_random_p=self.arg.mask_random_p,
            )

            # forward
            output, _, _ = self.model(data)
            loss = self.loss(output, gt)

            # backward
            self.optimizer.zero_grad()
            if self.arg.half:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            self.train_writer.add_scalar('total_loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log('\tMean training loss: {:.4f}. '.format(np.mean(loss_value)))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            step = 0

            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    N, C, T, V, M = data.shape
                    data = data.permute(0, 4, 2, 3, 1).contiguous().view(N*M*T, V, C)
                    data, gt = get_masked_input_and_labels(
                        data,
                        mask_value=self.arg.mask_value,
                        mask_p=self.arg.mask_p,
                        mask_remain_p=self.arg.mask_remain_p,
                        mask_random_p=self.arg.mask_random_p,
                    )

                    # forward
                    output, attns, _ = self.model(data)

                    if batch_idx == 0:
                        plot_attention_weights([attn[0].cpu() for attn in attns])
                        wandb.log({"Attentions": plt}, step=self.global_step)
                        plt.clf()

                    loss = self.loss(output, gt)

                    loss_value.append(loss.data.item())

                    step += 1

            self.eval_loss = np.mean(loss_value)

            if self.arg.phase == 'train':
                self.val_writer.add_scalar('total_loss', self.eval_loss, self.global_step)

            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                if epoch % self.arg.eval_freq == 0:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

                    torch.save(weights, os.path.join(self.arg.work_dir, 'recent.pt'))

                    if epoch > self.arg.save_epoch and self.eval_loss < self.best_loss:
                        self.best_loss = self.eval_loss
                        self.best_loss_epoch = epoch + 1
                        os.rename(os.path.join(self.arg.work_dir, 'recent.pt'),
                                  self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')



            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_loss_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
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
