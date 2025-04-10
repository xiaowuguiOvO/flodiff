# FloNa

Welcome to FloNa! This repository is the official implementation of paper "FloNa: Floor Plan Guided Embodied Visual Navigation".

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Dataset](#2-dataset)
- [3. Training](#4-training)
- [4. Testing in simulator](#5-testing-in-simulator)

## 1. Environment Setup
### a. Create a new `conda` environment
```bash
git clone https://github.com/GauleeJX/flodiff.git
cd flona
conda create -n flona -f environment.yaml
conda activate flona
```
### b. Install iGibson
```bash
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson
# conda create -y -n igibson python=3.8
# conda activate igibson
pip install -e . # This step takes about 4 minutes
```
### c. Download scenes
For our training and testing, we use Gibson static scenes—a dataset comprising over 500 reconstructions of homes and offices captured using a Matterport device. Please refer to this [link](https://stanfordvl.github.io/iGibson/dataset.html) to download the Gibson static scenes. We recommend saving the downloaded scenes in the following directory: /path_to_flona/iGibson/igibson/data/g_dataset.

## 2. Dataset
We collected navigation episodes on Gibson static scenes for training and evaluation. 
###  Download Collected Dataset from [BaiduDisk](https://pan.baidu.com/s/1kQnEJqHMPVRw0xcjGIUTvQ?pwd=skjj)
## 3. Training
```bash
cd /path_to_flona/training
python train.py
```
## 4. Testing in simulator

