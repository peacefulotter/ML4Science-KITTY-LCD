import os
import numpy as np
import torch.utils.data as data

from lcd.kitti.plots import plot_img_against_pc
from lcd.kitti.projection import __project
from lcd.kitti.dataset import KittiDataset
from lcd.kitti.preprocess import KittiPreprocess


root = './'
preprocess = KittiPreprocess(
    root=root,
    mode='debug'
)

preprocess[0]