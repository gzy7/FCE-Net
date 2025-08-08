# FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement

## Description
PyTorch version of FCE-Net.

## Experiment
### 1. Create Environment
We recommend using Conda to manage your Python environment.
```bash
# 1. Create and activate a new environment
conda create -n FCE python=3.9 -y
conda activate FCE
# 2. Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 2. Dataset Preparation
Paired Datasets (for supervised training)

[LOLv1](https://drive.google.com/drive/folders/1tzf-wDJrhUvh415msACOPRzm518guz81?usp=sharing)

[LOLv2](https://drive.google.com/drive/folders/1rLgUfFPcBCwO3oKJZZspiVobI6eFkx5E?usp=sharing)

Unpaired Datasets (for testing or unsupervised tasks)

[DICM](https://drive.google.com/drive/folders/19xLUJUv_-pCC8qhGPzUWagvVTn-eYEZc?usp=sharing)

[LIME](https://drive.google.com/drive/folders/1bwSTUbZpwMQCNbPFj_d7o9oMx4ZkVbA1?usp=sharing)

[MEF](https://drive.google.com/drive/folders/1q6P8skzFF1VAZhy63Yfx5sUYaoggHRkM?usp=sharing)

[NPE](https://drive.google.com/drive/folders/1LR5k8MD7fea9BfceqvveTsw4InQYjlXM?usp=sharing)

[VV](https://drive.google.com/drive/folders/1u6o9HORetJR_fhF8EwV5aD505P1wcUZL?usp=sharing)

### 3. Train
To train FCE-Net on the paired dataset:
```bash
python train.py
```

### 4. Test
After training, run the model on test images using:
```bash
python test.py
```
