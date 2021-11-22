# InfoGCN
Official PyTorch implementation of "InfoGCN: Representation Learning for Human Skeleton-based Action Recognition"

# Abstract
![image](src/framework.jpg)
Human skeleton-based action recognition offers a valuable means to understand the intricacies of human behavior because it can handle the complex relationships between physical constraints and intention. Although several studies have focused on encoding a skeleton, less attention has been paid to incorporating this information into the latent representations of human action. This paper proposes a learning framework for action recognition, InfoGCN, combining a novel learning objective and encoding method. First, we design an information bottleneck-based learning objective to guide the model to learn an informative but compact latent representation. To provide discriminative information for classifying action, we introduce attention-based graph convolution that captures the context-dependent intrinsic topology of human actions. In addition, we present a multi-modal representation of the skeleton using the relative position of joints, designed to provide complementary spatial information for joints. InfoGCN surpasses the known state-of-the-art on multiple skeleton-based action recognition benchmarks with the accuracy of 93.0\% on NTU RGB+D 60 cross-subject split, 89.8\% on NTU RGB+D 120 cross-subject split, and 97.0\% on NW-UCLA.

# Dependencies

- Python >= 3.6
- PyTorch >= 1.1.0
- NVIDIA Apex
- tqdm, tensorboardX

# Data Preparation

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

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
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



# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0
# Example: training provided baseline on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu120/csub/baseline --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn_bone --device 0
```

- To train model on NW-UCLA with bone or motion modalities, you need to modify `data_path` in `train_feeder_args` and `test_feeder_args` to "bone" or "motion" or "bone motion", and run

```
python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_xxx --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of CTRGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/csub/ctrgcn --bone-dir work_dir/ntu120/csub/ctrgcn_bone --joint-motion-dir work_dir/ntu120/csub/ctrgcn_motion --bone-motion-dir work_dir/ntu120/csub/ctrgcn_bone_motion
```

### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject [[Google Drive]](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing).
- Put files to <work_dir> and run **Testing** command to produce the final result.

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN), [HCN](https://github.com/huguyuehuhu/HCN-pytorch), and [Predict & Cluster](https://github.com/shlizee/Predict-Cluster).

Thanks to the original authors for their work!
