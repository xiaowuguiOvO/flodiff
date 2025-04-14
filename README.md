# FloNa

Welcome to FloNa! This repository is the official implementation of paper "FloNa: Floor Plan Guided Embodied Visual Navigation".

## Table of Contents
- [1. Environment Setup](#1-environment-setup)
- [2. Dataset](#2-dataset)
- [3. Training](#4-training)
- [4. Testing in simulator](#5-testing-in-simulator)

## 1. Environment Setup
### a. Create a new `conda` environment:
```bash
git clone https://github.com/GauleeJX/flodiff.git
cd flodiff
conda env create -n flodiff -f environment.yaml
conda activate flodiff
```
### b. Install the ```diffusion_policy```:
```bash
cd /path_to_flodiff
git clone https://github.com/real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```
## 2. Dataset
### a. Download our pre-collected dataset:
We collected navigation episodes on [Gibson static scenes](https://stanfordvl.github.io/iGibson/dataset.html) for training and evaluation. The structure of the episode data is organized as follows:
```bash
|--<train>
|   |--<scene_0>
|   |-----floor_plan.png        # the floor plan of the scene
|   |-----<traj0>               # one episode in the scene
|   |----------traj_0.npy           # a .npy file saving ground truth 2d position and 2d orientation for each frame
|   |----------traj_0.txt           # a .txt file saving ground truth 2D position and orientation for each frame
|   |----------traj_0.png           # a .png file showing the trajectory on the floor plan
|   |----------obs_0.png            # frame 0
|   |----------obs_1.png            # frame 1
|   |----------...
|   |-----<traj1>
|   |----------traj_1.npy
|   |----------traj_1.txt
|   |----------traj_1.png
|   |----------obs_0.png
|   |----------obs_1.png
|   |----------...
|   |-----...
|   |--<scene_1>
|   |-----...
|--<test>
|   |...
```
You can download it from [BaiduDisk](https://pan.baidu.com/s/1kQnEJqHMPVRw0xcjGIUTvQ?pwd=skjj). Please note that the dataset is approximately 500 GB, so make sure you have sufficient disk space.
### b. Unpack this tar archive:
```bash
cat /path_to_dataset/dataset.tar_* > /path_to_dataset/dataset.tar
tar -xvf dataset.tar -C /path_to_flodiff/
```
## 3. Training
```bash
cd /path_to_flodiff/training
python train.py
```
## 4. Testing in simulator
### a. Install iGibson
```bash
cd /path_to_flodiff
git clone https://github.com/StanfordVL/iGibson --recursive
cd iGibson
pip install -e . # This step takes about 4 minutes
```
### b. Download scenes
Please refer to this [link](https://stanfordvl.github.io/iGibson/dataset.html) to download the Gibson static scenes. We recommend saving the downloaded scenes in the following directory: /path_to_flodiff/iGibson/igibson/data/g_dataset.

### c. Testing 
You can 