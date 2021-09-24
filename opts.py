import argparse
import os
import random
import wandb
import numpy as np

from utils import str2bool

def get_pretrain_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='pretrain')

    parser.add_argument('--assume_yes', action='store_true', help='Say yes to every prompt')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--seed', type=int, default=random.randrange(200), help='random seed')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    parser.add_argument('--model_saved_name', default='')

    parser.add_argument('--log_dir', type=str, default='.', help='')
    parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--eval_start', type=int, default=1, help='The epoch number to start evaluating models')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    parser.add_argument('--feeder', default='feeders.feeder.Feeder', help='data loader will be used')
    parser.add_argument('--data_root', default='./data', help='data loader will be used')
    parser.add_argument('--train_dataset', default='ntu120', help='data loader will be used')
    parser.add_argument('--train_data_case', default='xsub', help='data loader will be used')
    parser.add_argument('--train_data_type', default='joint_align', help='data loader will be used')
    parser.add_argument('--test_dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--test_data_case', default='xsub', help='data loader will be used')
    parser.add_argument('--test_data_type', default='joint_align', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=os.cpu_count(), help='the number of worker for data loader')
    parser.add_argument('--train_ds_downsample', type=int, default=1, help='')
    parser.add_argument('--test_ds_downsample', type=int, default=1, help='')
    parser.add_argument('--num_point', type=int, default=25, help='')
    parser.add_argument('--num_person', type=int, default=2, help='')

    parser.add_argument('--model', default='model.port.MORT', help='the model will be used')
    parser.add_argument('--embd_dim', type=int, default=256, help='')
    parser.add_argument('--n_heads', type=int, default=8, help='')
    parser.add_argument('--n_layers', type=int, default=12, help='')
    parser.add_argument('--hidden_dim', type=int, default=1024, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    parser.add_argument('--half', action='store_true', help='Use half-precision (FP16) training')
    parser.add_argument('--amp_opt_level', type=int, default=1, help='NVIDIA Apex AMP optimization level')
    parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--device', type=int, default=[0, 1], nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--forward_batch_size', type=int, default=128, help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=50, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--optimizer_states', type=str, help='path of previously saved optimizer states')
    parser.add_argument('--checkpoint', type=str, help='path of previously saved training checkpoint')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--num_warmup_steps', type=int, default=2000, help='')

    parser.add_argument('--input_transform', type=str2bool, default=False, help='Transform input coordinate')
    parser.add_argument('--feature_transform', type=str2bool, default=False, help='Transform initial embeeding features')
    parser.add_argument('--vectorize', type=str2bool, default=False, help='vectorize joint location')

    parser.add_argument('--mask_p', type=float, default=0.15, help='')
    parser.add_argument('--loss', type=str, default='l2', help='')
    parser.add_argument('--mask_value', type=int, default=1, help='')
    parser.add_argument('--mask_random_p', type=float, default=0, help='')
    parser.add_argument('--mask_remain_p', type=float, default=0.1, help='')
    parser.add_argument('--lambda_motion', type=float, default=[1,1], nargs='+', help='')
    parser.add_argument('--lambda_trans_regularize', type=float, default=0.001, help='')
    parser.add_argument('--rot', action='store_true', help='')
    parser.add_argument('--max_rot', type=float, default=np.pi, help='')
    parser.add_argument('--lr_scheduler', type=str, default='step', help='')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='initial learning rate')

    parser.add_argument('--motion_time_steps', type=int, default=[], nargs='+', help='')
    parser.add_argument('--nonzero_mask', action='store_true', help='')
    parser.add_argument('--sinusoidal_pos_embds', action='store_true', help='')
    parser.add_argument('--jittering', action='store_true', help='')
    parser.add_argument('--crop_window', type=str2bool, default=False, help='Transform initial embeeding features')
    parser.add_argument('--localize', type=str2bool, default=False, help='localize joint location')

    return parser


def get_parser():
    # parameter priority: command line > config file > default
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument('--assume_yes', action='store_true', help='Say yes to every prompt')
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--seed', type=int, default=random.randrange(200), help='random seed')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--save_score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    parser.add_argument('--log_dir', type=str, default='.', help='')
    parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save_interval', type=int, default=0, help='the interval for storing models (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=1, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--eval_start', type=int, default=1, help='The epoch number to start evaluating models')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')

    parser.add_argument('--feeder', default='feeders.feeder.Feeder', help='data loader will be used')
    parser.add_argument('--data_root', default='./data', help='data loader will be used')
    parser.add_argument('--train_dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--train_data_case', default='xsub', help='data loader will be used')
    parser.add_argument('--train_data_type', default='joint_align', help='data loader will be used')
    parser.add_argument('--test_dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--test_data_case', default='xsub', help='data loader will be used')
    parser.add_argument('--test_data_type', default='joint_align', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=os.cpu_count(), help='the number of worker for data loader')
    parser.add_argument('--train_ds_downsample', type=int, default=1, help='')
    parser.add_argument('--test_ds_downsample', type=int, default=1, help='')
    parser.add_argument('--num_class', type=int, default=60, help='')
    parser.add_argument('--num_point', type=int, default=25, help='')
    parser.add_argument('--num_person', type=int, default=2, help='')

    parser.add_argument('--model', default='SPORT', help='the model will be used')
    parser.add_argument('--embd_dim', type=int, default=128, help='')
    parser.add_argument('--n_heads', type=int, default=8, help='')
    parser.add_argument('--n_layers', type=int, default=4, help='')
    parser.add_argument('--hidden_dim', type=int, default=512, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--freeze_port', action='store_true', help='')
    parser.add_argument('--pretrain_weight', type=str, default=None, help='')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    parser.add_argument('--half', action='store_true', help='Use half-precision (FP16) training')
    parser.add_argument('--amp_opt_level', type=int, default=1, help='NVIDIA Apex AMP optimization level')
    parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=128, help='training batch size')
    parser.add_argument('--forward_batch_size', type=int, default=128, help='Batch size during forward pass, must be factor of --batch-size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--optimizer_states', type=str, help='path of previously saved optimizer states')
    parser.add_argument('--checkpoint', type=str, help='path of previously saved training checkpoint')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--lr_scheduler', type=str, default='step', help='')
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='')

    parser.add_argument('--rot', action='store_true', help='')
    parser.add_argument('--max_rot', type=float, default=np.pi, help='')
    parser.add_argument('--nonzero_mask', action='store_true', help='')
    parser.add_argument('--jittering', action='store_true', help='')

    parser.add_argument('--input_transform', type=str2bool, default=False, help='Transform input coordinate')
    parser.add_argument('--feature_transform', type=str2bool, default=False, help='Transform initial embeeding features')
    parser.add_argument('--motion_time_steps', type=int, default=[], nargs='+', help='')
    parser.add_argument('--crop_window', type=str2bool, default=False, help='Transform initial embeeding features')
    parser.add_argument('--vectorize', type=str2bool, default=False, help='vectorize joint location')
    parser.add_argument('--localize', type=str2bool, default=False, help='localize joint location')
    return parser

