
import os
from dataset import KittiDataset

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
dataset = KittiDataset(root=root, mode='debug')

for idx in range(0, 1000, 1):
    seq_i, img_i, sample = dataset.map_index(idx)
    print(idx, " ->  (", seq_i, ", ", img_i, ", ", sample, ")")
