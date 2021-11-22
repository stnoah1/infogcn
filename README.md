# InfoGCN
Official PyTorch implementation of "InfoGCN: Representation Learning for Human Skeleton-based Action Recognition"

## Abstract
<img src="resources/main_fig.png" width="600" />
Human skeleton-based action recognition offers a valuable means to understand the intricacies of human behavior because it can handle the complex relationships between physical constraints and intention. Although several studies have focused on encoding a skeleton, less attention has been paid to incorporating this information into the latent representations of human action. This paper proposes a learning framework for action recognition, InfoGCN, combining a novel learning objective and encoding method. First, we design an information bottleneck-based learning objective to guide the model to learn an informative but compact latent representation. To provide discriminative information for classifying action, we introduce attention-based graph convolution that captures the context-dependent intrinsic topology of human actions. In addition, we present a multi-modal representation of the skeleton using the relative position of joints, designed to provide complementary spatial information for joints. InfoGCN surpasses the known state-of-the-art on multiple skeleton-based action recognition benchmarks with the accuracy of 93.0\% on NTU RGB+D 60 cross-subject split, 89.8\% on NTU RGB+D 120 cross-subject split, and 97.0\% on NW-UCLA.

## Dependencies

- Python >= 3.6
- PyTorch >= 1.1.0
- NVIDIA Apex
- tqdm, tensorboardX

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
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

## Training & Testing

### Training

- This is an exmaple command for training InfoGCN on NTU RGB+D 60 Cross Subject split. Please change the arguments if you want to customize the training. `--modal_idx` indicates k value of k-th mode represenation of skeleton. If you set `--use_vel=True`, the model will be trained with motion.

```
python main.py --model=SAGCN --half=True --batch_size=256 --test_batch_size=256 \
    --step 90 100 --num_epoch=110 --n_heads=3 --num_worker=4 --modal_idx=0 \
    --dataset=ntu --num_class=60 --lambda_1=1e-4 --lambda_2=1e-1 --z_prior_gain=3 \
    --save_epoch=60 --use_vel=False --datacase=CS --weight_decay=0.0005 \
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --model=SAGCN --half=True --test_batch_size=256 --n_heads=3 --num_worker=4 \
    --modal_idx=0 --dataset=ntu --num_class=60 --use_vel=False --datacase=CS \
    --phase=test --save_score=True --weights=<work_dir>/files/best_score.pt
```

- To ensemble the results of different modalities, run the following command:
```
python ensemble.py \
   --dataset=ntu/xsub \
   --position_ckpts \
   <work_dir_1>/files/best_score.pkl \
   <work_dir_2>/files/best_score.pkl \
   --motion_ckpts \
   <work_dir_3>/files/best_score.pkl \
   <work_dir_4>/files/best_score.pkl \

```

<!-- ### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject [[Google Drive]](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing).
- Put files to <work_dir> and run **Testing** command to produce the final result.
 -->
## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN), [HCN](https://github.com/huguyuehuhu/HCN-pytorch), and [Predict & Cluster](https://github.com/shlizee/Predict-Cluster).

Thanks to the original authors for their work!
