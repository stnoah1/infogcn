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
            npz_data = np.load('./data/ntu120/CSet_aligned.npz')
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

def ensemble_ntu60_cs():
    items = [
        ('./wandb/run-20211021_152338-3tne14dx/files/best_score.pkl', 1.5), # 90.5%, modal_idx=7, pose, random_rot
        ('./wandb/run-20211022_041137-3ooojndm/files/best_score.pkl', 1.5), # 90.5%, modal_idx=6, pose, random_rot
        # ('./wandb/run-20211022_090145-158l8k1u/files/best_score.pkl', 1.5), # 90.2%, modal_idx=5, pose, random_rot
        ('./wandb/run-20211021_042926-1ea2za8d/files/epoch1_test_score.pkl', 1.5), # 89.8%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211021_195035-1xd1m0iq/files/best_score.pkl', 1), # 88.6%, modal_idx=7, vel, random_rot
        ('./wandb/run-20211022_111250-3heeg9w8/files/best_score.pkl', 1), # 88.7, modal_idx=6, vel, random_rot
        # ('./wandb/run-20211021_111209-1j072ad2/files/epoch1_test_score.pkl', 1), # 88.5%, modal_idx=5, vel, random_rot(x)
        ('./wandb/run-20211021_135605-46q24qws/files/epoch1_test_score.pkl', 1), # 88.9%, modal_idx=0, vel, random_rot(x)
    ]

    ensemble('ntu/xsub', items)

def ensemble_ntu60_cv():
    items = [
        ('./wandb/run-20211021_084222-343ssdgx/files/epoch1_test_score.pkl', 1.5), # 95.5%, modal_idx=7, pose, random_rot (x)
        ('./wandb/run-20211022_132321-25wce9mh/files/best_score.pkl', 1.5), # 95.5%, modal_idx=6, pose, ramdom_rot
        # ('./wandb/run-20211021_174320-3poxff45/files/best_score.pkl', 1.5), # 95.3%, model_idx=5, pose, random_rot (x)
        ('./wandb/run-20211022_041136-1x63lf4o/files/best_score.pkl', 1.5), # 95.2%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211021_195555-18yn6ma4/files/epoch1_test_score.pkl', 1), # 93.6%, modal_idx=7, vel, random_rot (x)
        ('./wandb/run-20211022_172012-3inzut0k/files/best_score.pkl', 1), #93.9%, modal_idx=6, vel, random_rot
        # ('./wandb/run-20211022_234021-32k7vxbc/files/best_score.pkl', 1), #94.0%, modal_idx=5, vel, random_rot
        ('./wandb/run-20211022_041134-j2k33m98/files/best_score.pkl', 1), # 94.2%, modal_idx=0, vel, random_rot
    ]

    ensemble('ntu/xview', items)

def ensemble_ntu120_cs():
    items = [
    ]
    ensemble('ntu120/xsub', items)

def ensemble_ntu120_cset(): # alpha = 1e-3
    items = [
        ('./wandb/run-20211023_055740-1d58qxoa/files/best_score.pkl', 1.5), # 88.2%, modal_idx=6, pose, random_rot
        ('./wandb/run-20211023_024235-1g4jwn6g/files/best_score.pkl', 1.5), # 86.4%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211023_162158-3r8u9pf9/files/best_score.pkl', 1),   # 84.3%, modal_idx=7, vel, random_rot
        # ('./wandb/run-20211023_162341-1iu7ck41/files/best_score.pkl', 1),   # 84.4%, modal_dix=6, vel, random_rot
        ('./wandb/run-20211023_162359-23l8mw9k/files/best_score.pkl', 1),   # 84.4%, modal_dix=0, vel, random_rot
    ]
    ensemble('ntu120/xset', items)

if __name__ == "__main__":
    print('NTU60 XSub')
    ensemble_ntu60_cs()
    print('NTU60 XView')
    ensemble_ntu60_cv()
    # print('NTU60 XSub')
    # ensemble_ntu120_cs()
    print('NTU120 XSet')
    ensemble_ntu120_cset()

