# Code for **From Node to Graph: Joint Reasoning on Visual-Semantic Relational Graph for Zero-Shot Detection** 

## Installation
GRAN is built on MMDetection. You can follow the instructions below to install the dependencies and build GRAN.

## Dependencies
+ Linux with Python 3.6
+ PyTorch 1.1
+ torchvision 0.3
+ GCC 5.4
+ Cython: `pip install cython`
+ Dependencies: `pip install -r requirements.txt`

## Build
``` 
python setup.py develop
```

## Data Preparation
Download the annotations from the drive [link](https://drive.google.com/drive/folders/1NMtL_bbDJPHVrBkmvJKo6IYut8yJ463N?usp=sharing), put files to
```
data/coco/annotations/
```

Download MSCOCO 2014 dataset and unzip the images to the folders： 
```
data/coco/train2014/
data/coco/val2014/
```

## Usage
Training:
```
sh scripts/train_faster_sem_rcnn_r50_fpn_65_15_with_matcher_dist_train.sh
```
Inference:
```
sh scripts/inference_gzsd_65_15.sh
```
Evaluate:
```
sh scripts/eval_gzsd_65_15.sh
```