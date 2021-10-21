import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

def ensemble(ds, ckpt_dirs, alphas):
    if 'ntu120' in ds:
        num_class=120
        if 'xsub' in ds:
            npz_data = np.load('./data/ntu120/CSub_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in ds:
            npz_data = np.load('./data/ntu120/CSet_aligend.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in ds:
        num_class=60
        if 'xsub' in ds:
            npz_data = np.load('./data/ntu/CS_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in ds:
            npz_data = np.load('./data/ntu/CV_aligned.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    ckpts = []
    for ckpt_dir in ckpt_dirs:
        with open(os.path.join(ckpt_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            ckpts.append(list(pickle.load(f).items()))

    right_num = total_num = right_num_5 = 0

    for i in tqdm(range(len(label))):
        l = label[i]
        r = np.zeros(num_class)
        for alpha, ckpt in zip(alphas, ckpts):
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


def ensemble_ntu60_cv():
    ckpt_dirs = [
        './wandb/run-20211021_042946-1klfr5ej/files', # 95.0%, modal_idx=0, pose
        './wandb/run-20211021_084222-343ssdgx/files', # 95.5%, modal_idx=7, pose
        './wandb/run-20211021_134505-zkmgkh6u/files', # 94.0%, modal_idx=0, vel
    ]
    alphas = [1.5, 1.5, 1]

    ensemble('ntu/xview', ckpt_dirs, alphas)

def ensemble_ntu60_cs():
    ckpt_dirs = [
        './wandb/run-20211021_042926-1ea2za8d/files', # 89,8%, modal_idx=0, pose, random_rot
        './wandb/run-20211020_035202-2zjkchh8/files', # 90.2%, modal_idx=7, pose
        './wandb/run-20211021_135605-46q24qws/files', # 88.9%, modal_idx=0, vel, epoch=130
    ]
    alphas = [1.5, 1.5, 1]

    ensemble('ntu/xsub', ckpt_dirs, alphas)

if __name__ == "__main__":
    # ensemble_ntu60_cv()
    ensemble_ntu60_cs()
