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

def ensemble_ntu60_cs(r=1.5):
    print('NTU60 XSub')
    items = [
        ('./wandb/run-20211117_131331-21dmt7g5/files/best_score.pkl', r), # 90.7, modal_idx=7
        ('./wandb/run-20211022_041137-3ooojndm/files/best_score.pkl', r), # 90.5%, modal_idx=6, pose, random_rot
        ('./wandb/run-20211021_042926-1ea2za8d/files/epoch1_test_score.pkl', r), # 89.8%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211021_195035-1xd1m0iq/files/best_score.pkl', 1), # 88.6%, modal_idx=7, vel, random_rot
        ('./wandb/run-20211022_111250-3heeg9w8/files/best_score.pkl', 1), # 88.7, modal_idx=6, vel, random_rot
        ('./wandb/run-20211021_135605-46q24qws/files/epoch1_test_score.pkl', 1), # 88.9%, modal_idx=0, vel, random_rot(x)
    ]
    # items = [
        # ('./wandb/run-20211021_152338-3tne14dx/files/best_score.pkl', 1.5), # 90.5%, modal_idx=7, pose, random_rot
        # ('./wandb/run-20211021_042926-1ea2za8d/files/epoch1_test_score.pkl', 1.5), # 89.8%, modal_idx=0, pose, random_rot
        # ('./wandb/run-20211022_090145-158l8k1u/files/best_score.pkl', 1.5), # 90.2%, modal_idx=5, pose, random_rot
        # ('./wandb/run-20211021_111209-1j072ad2/files/epoch1_test_score.pkl', 1), # 88.5%, modal_idx=5, vel, random_rot(x)
    # ]
    ensemble('ntu/xsub', items)

def ensemble_ntu60_cv():
    print('NTU60 XView')
    items = [
        ('./wandb/run-20211021_084222-343ssdgx/files/epoch1_test_score.pkl', 1.5), # 95.5%, modal_idx=7, pose, random_rot (x)
        ('./wandb/run-20211022_132321-25wce9mh/files/best_score.pkl', 1.5), # 95.5%, modal_idx=6, pose, ramdom_rot
        ('./wandb/run-20211022_041136-1x63lf4o/files/best_score.pkl', 1.5), # 95.2%, modal_idx=0, pose, random_rot
        ('./wandb/run-20211021_195555-18yn6ma4/files/epoch1_test_score.pkl', 1), # 93.6%, modal_idx=7, vel, random_rot (x)
        ('./wandb/run-20211022_172012-3inzut0k/files/best_score.pkl', 1), #93.9%, modal_idx=6, vel, random_rot
        ('./wandb/run-20211022_041134-j2k33m98/files/best_score.pkl', 1), # 94.2%, modal_idx=0, vel, random_rot
    ]
    # items = [
        # ('./wandb/run-20211117_130121-4vttkfob/files/best_score.pkl', 1.5), # 95.5%, modal_idx=7, pose, new
        # ('./wandb/run-20211022_132321-25wce9mh/files/best_score.pkl', 1.5), # 95.5%, modal_idx=6, pose, ramdom_rot
        # ('./wandb/run-20211117_083251-2i4akl2o/files/best_score.pkl', 1.5), # 95.2%, modal_idx=0, pose, new
        # ('./wandb/run-20211021_195555-18yn6ma4/files/epoch1_test_score.pkl', 1), # 93.6%, modal_idx=7, vel, random_rot (x)
        # ('./wandb/run-20211022_172012-3inzut0k/files/best_score.pkl', 1), #93.9%, modal_idx=6, vel, random_rot
        # ('./wandb/run-20211022_041134-j2k33m98/files/best_score.pkl', 1), # 94.2%, modal_idx=0, vel, random_rot
        # ('./wandb/run-20211021_174320-3poxff45/files/best_score.pkl', 1.5), # 95.3%, model_idx=5, pose, random_rot (x)
        # ('./wandb/run-20211022_234021-32k7vxbc/files/best_score.pkl', 1), #94.0%, modal_idx=5, vel, random_rot
    # ]
    ensemble('ntu/xview', items)

def ensemble_ntu120_xsub(r=1.5):
    print('NTU120 XSub')
    items = [
        ('./wandb/run-20211116_010057-14wptl3p/files/best_score.pkl', r), #87.3%, modal_idx=7, pose
        # ('./wandb/run-20211116_005728-31ih3o9q/files/best_score.pkl', r), #86.5%, modal_idx=6, pose
        # ('./wandb/run-20211119_184301-9v3328uz/files/best_score.pkl', r), #85.3%, modal_idx=5, pose
        # ('./wandb/run-20211119_222108-3lvmocg9/files/best_score.pkl', r), #84.4%, modal_idx=4, pose
        # ('./wandb/run-20211120_102446-2zpbnaiq/files/best_score.pkl', r), #84.7%, modal_idx=3, pose
        # ('./wandb/run-20211119_222107-1332f221/files/best_score.pkl', r), #84.7%, modal_idx=2, pose
        # ('./wandb/run-20211120_202935-30ox3q0c/files/best_score.pkl', r), #84.7%, modal_idx=1, pose
        # ('./wandb/run-20211115_023846-3i8kb3lj/files/best_score.pkl', r), #85.1%, modal_idx=0, pose
        # ('./wandb/run-20211115_154208-3qdw6ycs/files/best_score.pkl', 1), # 82.2%, modal_idx=7, vel
        ('./wandb/run-20211115_154036-2inlr9np/files/best_score.pkl', 1), # 82.5%, modal_idx=6, vel
        # ('./wandb/run-20211120_061204-2vzkpveu/files/best_score.pkl', 1), # 82.2%, modal_idx=5, vel
        # ('./wandb/run-20211120_144311-2bgz98e1/files/best_score.pkl', 1), # 81.8%, modal_idx=4, vel
        # ('./wandb/run-20211119_222108-3lvmocg9/files/best_score.pkl', 1), # 82.2%, modal_idx=3, vel
        # ('./wandb/run-20211120_061631-263co39c/files/best_score.pkl', 1), # 81.9%, modal_idx=2, vel
        # ('./wandb/run-20211121_042448-xphciuhy/files/best_score.pkl', 1), # 82.0%, modal_idx=1, vel
        # ('./wandb/run-20211115_154037-2e6j8qc8/files/best_score.pkl', 1), # 82.1%, modal_idx=0, vel
    ]
    ensemble('ntu120/xsub', items)

# def ensemble_ntu120_cset(): # alpha = 1e-3
    # items = [
        # ('./wandb/run-20211024_203402-1msd7rcd/files/best_score.pkl', 1.5), # 88.5%, modal_idx=7, pose, random_rot, alpha=1e-1
        # ('./wandb/run-20211025_065304-3msq6zja/files/best_score.pkl', 1.5), # 88.2%, modal_idx=6, pose, random_rot, alpha=1e-1
        # # ('./wandb/run-20211025_065306-1lk374mv/files/best_score.pkl', 1.5), # 87.6%, modal_idx=5, pose, random_rot, alpha=1e-1
        # # ('./wandb/run-20211024_203401-3kx1oejz/files/best_score.pkl', 1.5), # 86.3%, modal_idx=0, pose, random_rot, alpha=1e-1
        # # ('./wandb/run-20211024_203403-1lwarbvs/files/best_score.pkl', 1), # 84.8%, modal_idx=7, vel, random_rot, alpha=1e-1
        # # ('./wandb/run-20211025_065307-3ehz8v80/files/best_score.pkl', 1), # 84.4% modal_idx=6, vel, random_rot, alpha=1e-1
        # # ('./wandb/run-20211025_065307-3lg8eqi5/files/best_score.pkl', 1), # 83.8%, modal_idx=5, vel, random_rot, alpha=1e-1
        # # ('./wandb/run-20211024_203404-2ek60kui/files/best_score.pkl', 1), # 84.4%, modal_idx=0, vel, random_rot, alpha=1e-1
    # ]
        # # ('./wandb/run-20211024_051020-2v3vm55l/files/best_score.pkl', 1.5), # 88.1%, modal_idx=7, pose, random_rot, alpha=1e-3
        # # ('./wandb/run-20211023_055740-1d58qxoa/files/best_score.pkl', 1.5), # 88.2%, modal_idx=6, pose, random_rot, alpha=1e-3
        # # ('./wandb/run-20211023_024235-1g4jwn6g/files/best_score.pkl', 1.5), # 86.4%, modal_idx=0, pose, random_rot, alpha=1e-3
        # # ('./wandb/run-20211023_162158-3r8u9pf9/files/best_score.pkl', 1),   # 84.3%, modal_idx=7, vel, random_rot, alpha=1e-3
        # # ('./wandb/run-20211023_162341-1iu7ck41/files/best_score.pkl', 1),   # 84.4%, modal_dix=6, vel, random_rot, alpha=1e-3
        # # ('./wandb/run-20211023_162359-23l8mw9k/files/best_score.pkl', 1),   # 84.4%, modal_dix=0, vel, random_rot, alpha=1e-3
    # ensemble('ntu120/xset', items)

def ensemble_ntu120_xset(): # alpha = 1e-3
    print('NTU120 XSet')
    items = [
        ('./wandb/run-20211024_203402-1msd7rcd/files/best_score.pkl', 1.5), # 88.5%, modal_idx=7, pose, random_rot, alpha=1e-1
        ('./wandb/run-20211025_065304-3msq6zja/files/best_score.pkl', 1.5), # 88.2%, modal_idx=6, pose, random_rot, alpha=1e-1
        ('./wandb/run-20211024_203401-3kx1oejz/files/best_score.pkl', 1.5), # 86.3%, modal_idx=0, pose, random_rot, alpha=1e-1
        ('./wandb/run-20211024_203403-1lwarbvs/files/best_score.pkl', 1), # 84.8%, modal_idx=7, vel, random_rot, alpha=1e-1
        ('./wandb/run-20211025_065307-3ehz8v80/files/best_score.pkl', 1), # 84.4% modal_idx=6, vel, random_rot, alpha=1e-1
        ('./wandb/run-20211024_203404-2ek60kui/files/best_score.pkl', 1), # 84.4%, modal_idx=0, vel, random_rot, alpha=1e-1
    ]
    # items = [
        # # ('./wandb/run-20211116_214456-80u1w1nq/files/best_score.pkl', 1.5), # 88.2 modal_idx=7, new
        # ('./wandb/run-20211116_143039-1y2e3hhu/files/best_score.pkl', 1.5), # 88.3 modal_idx=6, new
        # # ('./wandb/run-20211116_140706-44g4io0n/files/best_score.pkl', 1.5), # 86.4%, modal_idx=0, pose, new
        # ('./wandb/run-20211025_065306-1lk374mv/files/best_score.pkl', 1.5), # 87.6%, modal_idx=5, pose, random_rot, alpha=1e-1
        # ('./wandb/run-20211025_065307-3lg8eqi5/files/best_score.pkl', 1), # 83.8%, modal_idx=5, vel, random_rot, alpha=1e-1

        # # ('./wandb/run-20211024_203403-1lwarbvs/files/best_score.pkl', 1), # 84.8%, modal_idx=7, vel, random_rot, alpha=1e-1
        # # ('./wandb/run-20211116_213656-1e67fvxl/files/best_score.pkl', 1), # 84.3% modal_idx=6, vel, new
        # # ('./wandb/run-20211116_211700-3rugmd3j/files/best_score.pkl', 1), # 84.2%, modal_idx=0, vel,new
    # ]
    ensemble('ntu120/xset', items)

if __name__ == "__main__":
    # ensemble_ntu60_cs()
    # ensemble_ntu60_cv()
    ensemble_ntu120_xsub()
    # ensemble_ntu120_xset()

