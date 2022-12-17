
import os
from eval_dataset import KittiEvalDataset
from descriptors import find_descriptors_correspondence

import sys
sys.path.append('../')
from models.patchnet import PatchNetAutoencoder
from models.pointnet import PointNetAutoencoder

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
dataset = KittiEvalDataset(root)

device = 'cpu'
patchnet = PatchNetAutoencoder(256, True)
pointnet = PointNetAutoencoder(256,6,6,True)
patchnet.to(device)
pointnet.to(device)

idx = 0
batch = dataset[idx]
x = [x.to(device).float() for x in batch]

pred_pcs, point_descriptors = pointnet(x[0])
pred_imgs, patch_descriptors = patchnet(x[1])

correspondences = find_descriptors_correspondence(point_descriptors, patch_descriptors)
print(correspondences.shape)

