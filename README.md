# FCE-Net: A Frequency-Color-Edge Aware YUV Network for Low-Light Image Enhancement

## Description
PyTorch version of FCE-Net.

## Experiment
### 1. Create Environment
<pre> conda create -n FCE python=3.9 -y

conda activate FCE

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia </pre>

### 2.Prepare Datasets
Paired Datasets

[LOLv1](https://drive.google.com/drive/folders/1tzf-wDJrhUvh415msACOPRzm518guz81?usp=drive_link)

[LOLv2](https://drive.google.com/drive/folders/1rLgUfFPcBCwO3oKJZZspiVobI6eFkx5E?usp=drive_link)

Unpaired Datasets

[DICM](https://drive.google.com/drive/folders/19xLUJUv_-pCC8qhGPzUWagvVTn-eYEZc?usp=drive_link)

[LIME](https://drive.google.com/drive/folders/1bwSTUbZpwMQCNbPFj_d7o9oMx4ZkVbA1?usp=drive_link)

[MEF](https://drive.google.com/drive/folders/1q6P8skzFF1VAZhy63Yfx5sUYaoggHRkM?usp=drive_link)

[NPE](https://drive.google.com/drive/folders/1LR5k8MD7fea9BfceqvveTsw4InQYjlXM?usp=drive_link)

[VV](https://drive.google.com/drive/folders/1u6o9HORetJR_fhF8EwV5aD505P1wcUZL?usp=drive_link)

### 3.Train
```python train.py```

### 4.Test
```python test.py```
