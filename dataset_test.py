import os
import numpy as np
import torch.utils.data as data

from lcd.kitti.plots import plot_img_against_pc
from lcd.kitti.projection import __project
from lcd.kitti.dataset import KittiDataset
from lcd.kitti.preprocess import KittiPreprocess

from lcd.models.patchnet import PatchNetAutoencoder
from lcd.models.pointnet import PointNetAutoencoder

root = os.path.join(os.path.dirname(os.path.realpath(__file__)))
dataset = KittiDataset(root=root, mode='debug')

"""pc, img = dataset[199]
pc, colors = np.hsplit(pc, 2)
plots.plot_pc(pc, colors)"""

loader = data.DataLoader(
    dataset,
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
)
device = 'cpu'
patchnet = PatchNetAutoencoder(256, True)
pointnet = PointNetAutoencoder(256,6,6,True)
patchnet.to(device)
pointnet.to(device)

for i, x in enumerate([dataset[i] for i in range(100, 110, 2)]):
    pp = KittiPreprocess(root, 'debug')
    img_folder, pc_folder, seq_i, img_i, cam_i = pp.dataset[i]
    img, pc, intensity, sn = pp.load_item(img_folder, pc_folder, img_i)

    pc = x[0].T[:3]
    pts_in_frame, _ = __project(pp.calibs, pc, 0, 2)

    img = x[1]
    print(img.shape, img.dtype,  pts_in_frame.shape)
    min_h = np.floor(np.min(pts_in_frame[:, 0]))
    min_w = np.floor(np.min(pts_in_frame[:, 1]))
    # img += np.array([min_h, min_w, 0], dtype=np.uint8)
    
    plot_img_against_pc(img, pts_in_frame, z=None)

    print("input batch", x[0].shape, x[1].shape)