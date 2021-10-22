import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

def ensemble(ds, items):
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

    ckpt_dirs, alphas = list(zip(*items))

    ckpts = []
    for ckpt_dir in ckpt_dirs:
        with open(ckpt_dir, 'rb') as f:
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
    items = [
        ('./wandb/run-20211021_084222-343ssdgx/files/epoch1_test_score.pkl', 1.5), # 95.5%, modal_idx=7, pose
        ('./wandb/run-20211021_042946-1klfr5ej/files/epoch1_test_score.pkl', 1.5), # 95.0%, modal_idx=0, pose
        ('./wandb/run-20211021_174320-3poxff45/files/best_score.pkl', 1.5), # 95.3%,  model_idx=5, pose
        ('./wandb/run-20211021_195555-18yn6ma4/files/epoch1_test_score.pkl', 1), # 93.6%, modal_idx=7, vel
        ('./wandb/run-20211021_134505-zkmgkh6u/files/epoch1_test_score.pkl', 1), # 94.0%, modal_idx=0, vel
    ]

    ensemble('ntu/xview', items)

def ensemble_ntu60_cs():
    items = [
        ('./wandb/run-20211021_152338-3tne14dx/files/best_score.pkl',        1.5), # 90.5%, modal_idx=7, pose, random_rot
        ('./wandb/run-20211021_235149-2g8ul0tg/files/epoch1_test_score.pkl', 1.5), # 89.7 modal_idx=6, pose
        ('./wandb/run-20211021_042926-1ea2za8d/files/epoch1_test_score.pkl', 1.5), # 89.8%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211021_175515-3pr6bb8c/files/best_score.pkl', 1), # 88.8%, modal_idx=6, vel
        ('./wandb/run-20211021_111209-1j072ad2/files/epoch1_test_score.pkl', 1), # 88.5%, modal_idx=5, vel
        ('./wandb/run-20211021_135605-46q24qws/files/epoch1_test_score.pkl', 1), # 88.9%, modal_idx=0, vel, epoch=130
        # ('./wandb/run-20211020_035202-2zjkchh8/files/epoch1_test_score.pkl', 1), # 90.2%, modal_idx=7, pose

    ]

    ensemble('ntu/xsub', items)

if __name__ == "__main__":
    print('NTU60 CV')
    ensemble_ntu60_cv()
    print('NTU60 CS')
    ensemble_ntu60_cs()
