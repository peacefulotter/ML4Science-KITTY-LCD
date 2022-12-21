import os
import torch
import numpy as np
import open3d as o3d
import torch.utils.data as data

from .preprocess import KittiPreprocess
from .pointcloud import downsample_neighbors
from .patches import extract_patches_2d
from .projection import project
from .poses import import_poses

class KittiEvalDataset(data.Dataset):
    def __init__(self, root, patch_w=64, patch_h=64, min_pc=32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.patch_size = (patch_h, patch_w)
        self.min_pc = min_pc
        self.preprocess = KittiPreprocess(root, 'debug')
        self.poses = import_poses(root, self.preprocess.seq_list)
        print('--------- KittiEvalDataset init Done ---------')
        
    def get_extracted_pose(self, seq_i, img_i):
        pose = self.poses[seq_i][img_i]
        R = pose[:3, :3]
        t = pose[:, 3]
        return R, t

    def intrinsic_params(self, seq_i, cam_i):
        return self.preprocess.calibs[seq_i][f'K{cam_i}']

    # increased voxel_grid_size as in training
    def voxel_down_sample(self, pc, voxel_grid_size=10):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_size)
        return np.asarray(down_pcd.points)
        
    def get_neighbourhoods(self, pc, img, seq_i, cam_i):
        print(pc.shape, img.shape, seq_i, cam_i)
        _, depth_mask, in_image_mask, colors = project(self.preprocess.calibs, pc, img, seq_i, cam_i)
        total_mask = self.preprocess.combine_masks(depth_mask, in_image_mask)
        pc = pc.T[total_mask]
        ds_pc = self.voxel_down_sample(pc)
        neighbors_indices, _ = downsample_neighbors(ds_pc, pc, self.min_pc)
        pc = np.c_[ pc, colors ]
        return pc[neighbors_indices]

    def get_patches(self, img):
        # TODO: remove max_patches
        return extract_patches_2d(
            img / 255, self.patch_size, max_patches=0.0001
        )

    @staticmethod
    def collate(batch):
        pcs = None
        imgs = None
        origins = None
        details = torch.zeros((len(batch), 3), dtype=int)
        for i, (pc, img, origin, dets) in enumerate(batch):
            if pcs is None:
                pcs = pc
                imgs = img
                origins = origin
            else:
                pcs = np.r_[ pcs, pc ]
                imgs = np.r_[ imgs, img ]
                origins = np.r_[ origins, origin ]
            details[i] = torch.tensor(dets)
        pcs = torch.from_numpy(pcs)
        imgs = torch.from_numpy(imgs)
        origins = torch.from_numpy(origins)
        return pcs, imgs, origins, details

    def __len__(self):
        return len(self.preprocess)

    def __getitem__(self, index):
        img_folder, pc_folder, seq_i, img_i, cam_i = self.preprocess.dataset[index]
        img, pc, _, _ = self.preprocess.load_item(img_folder, pc_folder, img_i)
        patches, origins = self.get_patches(img)
        neighbourhoods = self.get_neighbourhoods(pc, img, seq_i, cam_i)
        return neighbourhoods, patches, origins, [seq_i, img_i, cam_i]


if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    dataset = KittiEvalDataset(root=root)
    neighborhoods, patches = dataset[0]
    print(neighborhoods.shape, patches.shape)