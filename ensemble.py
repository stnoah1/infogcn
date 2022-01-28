import argparse
import pickle
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
    elif 'ucla' in ds:
        num_class=10
        npz_data = np.load('./data/ntu/CS_aligned.npz')
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')

    parser.add_argument('--position_ckpts', nargs='+',
                        help='Directory containing "epoch1_test_score.pkl" for position eval results')
    parser.add_argument('--motion_ckpts', nargs='+',
                        help='Directory containing "epoch1_test_score.pkl" for motion eval results')

    arg = parser.parse_args()

    item = []
    for ckpt in arg.position_ckpts:
        item.append((ckpt, 1.5))
    for ckpt in arg.motion_ckpts:
        item.append((ckpt, 1))

    ensemble(arg.dataset, item)
