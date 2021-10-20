import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        nargs='+',
                        help='weighted summation',
                        type=float)

    parser.add_argument('--num-class',
                        default=1,
                        help='number of class in dataset',
                        type=int)

    parser.add_argument('--ckpt-dirs',
                        nargs='+',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')


    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/ntu120/CSub_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/ntu120/CSet_aligend.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/ntu/CS_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/ntu/CV_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    ckpts = []
    for ckpt_dir in arg.ckpt_dirs:
        with open(os.path.join(ckpt_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            ckpts.append(list(pickle.load(f).items()))

    right_num = total_num = right_num_5 = 0

    for i in tqdm(range(len(label))):
        l = label[i]
        r = np.zeros(arg.num_class)
        for alpha, ckpt in zip(arg.alpha, ckpts):
            _, r11 = ckpt[i]
            r += r11 * alpha

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
