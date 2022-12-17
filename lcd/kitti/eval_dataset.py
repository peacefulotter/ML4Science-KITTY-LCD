import os
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data
from sklearn.feature_extraction import image

from preprocess import KittiPreprocess
from pointcloud import downsample_neighbors

class KittiEvalDataset(data.Dataset):
    def __init__(self, root, patch_w=64, patch_h=64, min_pc=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.min_pc = min_pc
        self.preprocess = KittiPreprocess(root, 'debug') # TODO: change to test
        print('--------- KittiEvalDataset init Done ---------')
        
    # increased voxel_grid_size as in training
    def voxel_down_sample(self, pc, voxel_grid_size=0.2):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_size)
        return np.asarray(down_pcd.points)
        
    def get_neighborhoods(self, pc, img, seq_i, cam_i):
        _, depth_mask, in_image_mask, colors = self.preprocess.project_pointcloud(pc, img, seq_i, cam_i)
        total_mask = self.preprocess.combine_masks(depth_mask, in_image_mask)
        pc = pc.T[total_mask]
        ds_pc = self.voxel_down_sample(pc)
        neighbors_indices, _ = downsample_neighbors(ds_pc, pc, self.min_pc)
        pc = np.c_[ pc, colors ]
        return pc[neighbors_indices]

    def get_patches(self, img):
        patch_size = (self.patch_h, self.patch_w)
        # TODO: remove max_patches
        return image.extract_patches_2d(img / 255, patch_size, max_patches=0.01)

    def __len__(self):
        return len(self.preprocess)

    def __getitem__(self, index):
        img_folder, pc_folder, seq_i, img_i, cam_i = self.preprocess.dataset[index]
        img, pc, _, _ = self.preprocess.load_item(img_folder, pc_folder, img_i)
        patches = self.get_patches(img)
        neighborhoods = self.get_neighborhoods(pc, img, seq_i, cam_i)
        return torch.from_numpy(neighborhoods), torch.from_numpy(patches)


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    dataset = KittiEvalDataset(root=root)
    neighborhoods, patches = dataset[0]
    print(neighborhoods.shape, patches.shape)