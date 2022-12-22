# ML4Science - Training an LCD model on the KITTI dataset

## Table of contents
* [Abstract](#abstract)
* [Goal](#goal)
* [Installation](#installation)
* [How to use](#how-to-use)
* [Dependencies](#dependenciess)
* [Repository structure overview](#repository-structure-overview)
* [LCD outline](#lcd-outline)

## Abstract
In the interest of applying theoretical knowledge learned in the course "Machine Learning" at EPFL, this project was created and supervised by the Geodetic Engineering Laboratory of EPFL. We were given an inital proposition which may be found [here](https://www.epfl.ch/labs/topo/student_projects/aerial-2d-and-3d-vision-a-joint-deep-learning-assisted-application/ "here"). In particular, this project was supervised by Kyriaki Mouzakidou,  Jesse Lahaye and Aurélien Arnaud Brun of the Geodetic Engineering Laboratory of EPFL.

## Goal
The idea behind this project is to adapt the [LCD model](https://github.com/hkust-vgd/lcd "LCD model") to the [KITTI dataset](https://www.cvlibs.net/datasets/kitti/ "KITTI dataset") or the inverse. Once this is complete, the model can be trained and tested on this dataset so as to directly test its performance on a widely used dataset such as KITTI. Not only this, the LCD model has only currently been tested indoors so the varied scenarios of the KITTI dataset will serve as a good test for robustness.

## Installation
1. Clone the master branch
1. Install dependencies listed below
1. Download dataset from: todo ....


## How to use
### Local: master branch
Once inside the ML4Science-KITTI-LCD directory run these commands:

For training:
```bash
python kitti_train.py
```
This will automatically preprocess the KITTI data and save each sample of a sequence in the folder 'kitti_data' using subfolders to divide the data. I.e. The first sample for of the first sequence will be stored in "kitti_data/0/0.npz".

For testing
```bash
python kitti_eval.py
```
This will take the trained model and preprocessed data and then print the respective performance of each image patch and pointnet against the ground truth data.

## Dependencies
- numpy
- scipy
- pytorch
- [opencv](https://github.com/opencv/opencv-python "opencv"): Used for the PnP RANSAC algorithm to search for poses given image-point cloud pairs
- [open3d](https://github.com/isl-org/Open3D "open3d"): downsampling/manipulation/visualisation of point clouds
- [Faiss](https://github.com/facebookresearch/faiss "Faiss"): Used for quick and efficient euclidean distance comparison

## Repository structure overview
    ├── LCD                     # The LCD model, preprocessing, metrics, plotting
    ├── logs                    # Pretrained LCD model and training logs/settings
    ├── poses                   # ground truth rotation matrix and translation vector from kitti dataset
    ├── LICENSE
    ├── README.md
	├── __init__.py
    ├── config.json             # Original GPU/training settings of LCD
	├── config_kitti.json       # GPU/training settings of LCD used in this project
    ├── kitti_eval.py           # Script to evaluate the LCD model trained on KITTI
	├── kitti_train.py          # Script to train the LCD model on preprocessed KITTI
	├── preprocessing.sbatch    # SCITAS commands to preprocess KITTI
    ├── raw_kitti_proj.py
	├── requirements.txt        # SCITAS package dependencies
    └── training.sbatch         # SCITAS commands to train KITTI


### LCD outline
    ├── ...
    ├── LCD                     # The LCD model, preprocessing, metrics, plotting
    │   ├── kitti               # Preprocessing, plotting, and testing of KITTI dataset for LCD model
	│   │   ├── __init__.py
	│   │   ├── calib.py        # Parsing of KITTI calib files
	│   │   ├── dataset.py      # Training dataset class: Able to extract data for a given sequence, image and sample based on index
	│   │   ├── descriptors.py  # Pointnet vs Patchnet descriptor matching based on euclidean distance
	│   │   ├── eval_dataset.py # Evaluation dataset class
	│   │   ├── metrics.py      # Metric functions and pose estimation
	│   │   ├── patches.py      # Extract subpatches of a given size from an image
	│   │   ├── plots.py        # Various plotting functions to visualise data transformations
	│   │   ├── pointcloud.py   # Pointcloud downsampling and transformation
	│   │   ├── poses.py        # Parsing ground poses
	│   │   ├── preprocess.py   # Script to preprocess KITTI dataset for LCD training
	│   │   ├── pointcloud.py   # Pointcloud downsampling and transformation
	│   │   └── projection.py   # Project pointcloud into images (and extract colour) and filter useful points
    │   ├── losses              # Loss functions of LCD
    │   └── models              # LCD autoencoders: generate descriptors of size 256 from a given image and a given point cloud
	└── ...