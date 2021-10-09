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
    parser.add_argument('--noise-ratio', type=float, default=0.1, help='initial learning rate')

    # data
    parser.add_argument('--num_point', type=int, default=25, help='')
    parser.add_argument('--num_person', type=int, default=2, help='')
    parser.add_argument('--num_class', type=int, default=60, help='')
    parser.add_argument('--dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--datacase', default='CS', help='data loader will be used')
    parser.add_argument('--use-bone', type=str2bool, default=False, help='')
    parser.add_argument('--use-vel', type=str2bool, default=False, help='')


    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=25, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=8, help='the number of worker for data loader')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--n_heads', type=int, default=3, help='')

    # port
    parser.add_argument('--n-heads', type=int, default=8, help='')
    parser.add_argument('--embd-dim', type=int, default=64, help='')
    parser.add_argument('--n-layers', type=int, default=6, help='')
    parser.add_argument('--freeze-port', type=str2bool, default=True, help='')
    parser.add_argument('--pretrain-weight', help='')


    # optim
    parser.add_argument('--base-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[35, 55], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=64, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=65, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='weight decay for optimizer')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm-up-epoch', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=1.)

    # apex
    parser.add_argument('--half', action='store_true', help='Use half-precision (FP16) training')
    parser.add_argument('--amp-opt-level', type=int, default=1, help='NVIDIA Apex AMP optimization level')

    return parser


def get_pretrain_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser( description='Spatial Temporal Graph Convolution Network')
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode; default false')
    parser.add_argument('--log_dir', type=str, default='.', help='')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--eval-freq', type=int, default=10, help='')

    # data
    parser.add_argument('--num-point', type=int, default=25, help='')
    parser.add_argument('--num-person', type=int, default=2, help='')
    parser.add_argument('--num-class', type=int, default=60, help='')
    parser.add_argument('--dataset', default='ntu', help='data loader will be used')
    parser.add_argument('--datacase', default='CS', help='data loader will be used')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_ntu.Feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=8, help='the number of worker for data loader')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
    parser.add_argument('--n-heads', type=int, default=8, help='')
    parser.add_argument('--embd-dim', type=int, default=64, help='')
    parser.add_argument('--n-layers', type=int, default=6, help='')

    # mask
    parser.add_argument('--mask-p', type=float, default=0.3, help='')
    parser.add_argument('--mask-value', type=int, default=1, help='')
    parser.add_argument('--mask-random-p', type=float, default=0, help='')
    parser.add_argument('--mask-remain-p', type=float, default=0.1, help='')

    # optim
    parser.add_argument('--loss', default='l2', help='loss')
    parser.add_argument('--base-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[35, 55], nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='Adam', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=65, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0004, help='weight decay for optimizer')
    parser.add_argument('--num-warmup-steps', type=int, default=1000)

    # apex
    parser.add_argument('--half', action='store_true', help='Use half-precision (FP16) training')
    parser.add_argument('--amp-opt-level', type=int, default=1, help='NVIDIA Apex AMP optimization level')

    return parser

