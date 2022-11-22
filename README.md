# InfoGCN
Official PyTorch implementation of "[InfoGCN: Representation Learning for Human Skeleton-based Action Recognition](https://openaccess.thecvf.com/content/CVPR2022/html/Chi_InfoGCN_Representation_Learning_for_Human_Skeleton-Based_Action_Recognition_CVPR_2022_paper.html)", CVPR22.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/infogcn-representation-learning-for-human/skeleton-based-action-recognition-on-n-ucla)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-n-ucla?p=infogcn-representation-learning-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/infogcn-representation-learning-for-human/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=infogcn-representation-learning-for-human)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/infogcn-representation-learning-for-human/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=infogcn-representation-learning-for-human)

## Abstract
<img src="resources/main_fig.png" width="600" />
Human skeleton-based action recognition offers a valuable means to understand the intricacies of human behavior because it can handle the complex relationships between physical constraints and intention. Although several studies have focused on encoding a skeleton, less attention has been paid to embed this information into the latent representations of human action. InfoGCN proposes a learning framework for action recognition combining a novel learning objective and an encoding method. First, we design an information bottleneck-based learning objective to guide the model to learn informative but compact latent representations. To provide discriminative information for classifying action, we introduce attention-based graph convolution that captures the context-dependent intrinsic topology of human action. In addition, we present a multi-modal representation of the skeleton using the relative position of joints, designed to provide complementary spatial information for joints. InfoGCN surpasses the known state-of-the-art on multiple skeleton-based action recognition benchmarks with the accuracy of 93.0% on NTU RGB+D 60 cross-subject split, 89.8% on NTU RGB+D 120 cross-subject split, and 97.0% on NW-UCLA.

## Dependencies

- Python >= 3.6
- PyTorch >= 1.7.0
- NVIDIA Apex
- tqdm, tensorboardX, wandb

## Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from CTR-GCN repo: [https://github.com/Uason-Chen/CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame and vertically align to the ground
 python seq_transformation.py
```

## Training & Testing

### Training
- We set the seed number for Numpy and PyTorch as 1 for reproducibility.
- If you want to reproduce our works, please find the details in the supplementary matrials. The hyperparameter setting differs depending on the training dataset. 
- This is an exmaple command for training InfoGCN on NTU RGB+D 60 Cross Subject split. Please change the arguments if you want to customize the training. `--k` indicates k value of k-th mode represenation of skeleton. If you set `--use_vel=True`, the model will be trained with motion.

```
python main.py --half=True --batch_size=128 --test_batch_size=128 \
    --step 90 100 --num_epoch=110 --n_heads=3 --num_worker=4 --k=1 \
    --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
    --use_vel=False --datacase=NTU60_CS --weight_decay=0.0005 \
    --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --half=True --test_batch_size=128 --n_heads=3 --num_worker=4 \
    --k=1 --dataset=ntu --num_class=60 --use_vel=False --datacase=NTU60_CS \
    --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder \
    --phase=test --save_score=True --weights=<path_to_weight>
```

- To ensemble the results of different modalities, run the following command:
```
python ensemble.py \
   --dataset=ntu/xsub \
   --position_ckpts \
      <work_dir_1>/files/best_score.pkl \
      <work_dir_2>/files/best_score.pkl \
      ...
   --motion_ckpts \
      <work_dir_3>/files/best_score.pkl \
      <work_dir_4>/files/best_score.pkl \
      ...
```

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN), [HCN](https://github.com/huguyuehuhu/HCN-pytorch), and [Predict & Cluster](https://github.com/shlizee/Predict-Cluster).

Thanks to the original authors for their work!
