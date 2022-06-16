import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser( description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--log_dir', type=str, default='.', help='')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--noise_ratio', type=float, default=0.5, help='')

    # data
    parser.add_argument('--n_desired', type=int, default=40000, help='')
    parser.add_argument('--num_point', type=int, default=25, help='')
    parser.add_argument('--num_person', type=int, default=2, help='')
    parser.add_argument('--num_class', type=int, default=60, help='')
    parser.add_argument('--dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--datacase', default='CS', help='data loader will be used')
    parser.add_argument('--use_vel', type=str2bool, default=False, help='')


    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save_interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save_epoch', type=int, default=60, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--balanced_sampling', type=str2bool, default=False, help='the number of worker for data loader')
    parser.add_argument('--random_rot', type=str2bool, default=True, help='')
    parser.add_argument('--repeat', type=int, default=1, help='the number of repeat for data')

    # model
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--n_heads', type=int, default=3, help='')
    parser.add_argument('--k', type=int, default=0, help='')
    parser.add_argument('--z_prior_gain', type=int, default=3, help='')
    parser.add_argument('--graph', type=str, default='graph.ntu_rgb_d.Graph', help='')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[90, 100], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=110, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)
    parser.add_argument('--lambda_1', type=float, default=1e-4)
    parser.add_argument('--lambda_2', type=float, default=1e-1)

    # apex
    parser.add_argument('--half', type=str2bool, default=True, help='Use half-precision (FP16) training')
    parser.add_argument('--amp_opt_level', type=int, default=1, help='NVIDIA Apex AMP optimization level')

    return parser

