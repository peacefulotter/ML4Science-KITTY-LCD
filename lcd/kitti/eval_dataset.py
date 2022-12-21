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
    '''
    This is the evaluation dataset class to use the preprocessed data 
    from KittiPreprocess.

    This class is less straightforward then KittiDataset because evaluating
    the LCD model requires a bit more work. See __get_item__ for more informations.
    '''

    def __init__(self, root, patch_w=64, patch_h=64, min_pc=32, num_pc=1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.patch_size = (patch_h, patch_w)
        self.min_pc = min_pc
        self.num_pc = num_pc
        self.preprocess = KittiPreprocess(root, 'test')
        self.poses = import_poses(root, self.preprocess.seq_list)
        print('--------- KittiEvalDataset init Done ---------')
        
    def get_extracted_pose(self, seq_i, img_i):
        '''
        Extract the ground truth pose, i.e. rotation matrix and translation vector
        at the sequence seq_i and for image img_i.

        Returns:
        - (3, 3) R: Ground truth rotation matrix
        - (3, 1) t: Ground truth translation vector
        '''
        pose = self.poses[seq_i][img_i]
        R = pose[:3, :3]
        t = pose[:, 3]
        return R, t

    def intrinsic_params(self, seq_i, cam_i):
        '''
        Get the camera intrinsic parameters K for sequence seq_i and camera cam_i.

        Returns:
        - (3, 3) K: Camera intrinsic parameters
        '''
        return self.preprocess.calibs[seq_i][f'K{cam_i}']

    def voxel_down_sample(self, pc, voxel_grid_size=10):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        down_pcd = pcd.voxel_down_sample(voxel_size=voxel_grid_size)
        return np.asarray(down_pcd.points)
        
    def get_neighbourhoods(self, pc, img, seq_i, cam_i):
        '''
        Get the neighbourhoods in a 1m radius for each point in the given pointcloud.

        FIXME: The way this is done is not ideal because we are not supposed to know
        the projection matrix for testing. However, this is the only reasonable way 
        we thought of to avoid computing the neighbourhoods for ALL the points and also
        to get the RGB color per point. 

        Returns:
        - (M, num_pc, 6) M neighbourhoods of num_pc point each in 6d (xyzrgb) 
        '''
        _, depth_mask, in_image_mask, colors = project(self.preprocess.calibs, pc, img, seq_i, cam_i)
        total_mask = self.preprocess.combine_masks(depth_mask, in_image_mask)
        pc = pc.T[total_mask]
        ds_pc = self.voxel_down_sample(pc)
        neighbors_indices, _ = downsample_neighbors(
            ds_pc, pc, self.min_pc, radius=1, downsample=self.num_pc
        )
        pc = np.c_[ pc, colors ]
        return pc[neighbors_indices]

    def get_patches(self, img):
        '''
        Get the all the possible patches of size (patch_h, patch_w) in the 
        given image (img).

        FIXME: set max_patches to 1 to get all the possibles patches
        note that doing this increases multiplies the nb of patches by 100

        Params:
        - (H, W, 3) img: RGB Image 
        
        Returns:
        - (P, patch_h, patch_w, 3) P RGB patches of width patch_w and height patch_h
        '''
        return extract_patches_2d(
            img, self.patch_size, max_patches=0.01
        )

    @staticmethod
    def collate(batch):
        '''
        pytorch custom collate function.
        Used to concatenate multiple items (from __get_item__) into a single
        pytorch batch. This function is required we the torch dataloader is 
        dealing with unusual / more complex data format.

        This function simply concatenates the items in the given batch into
        a proper pytorch batch.
        '''
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
        '''
        1. Load the pointcloud and image at the corresponding seq_i
        sequence and image img_i.
        2. Get all the possible (patch_h, patch_w) patches in the image.
        3. Get all the possible neighbourhoods of size num_pc in the pointcloud pc.

        Returns:
        - (M, num_pc, 6) neighbourhoods: M neighbourhoods of num_pc point each in 6d (xyzrgb) 
        - (P, patch_h, patch_w, 3) patches: P RGB patches of width patch_w and height patch_h
        - (P, 2) origins: origin coordinates for each patch in patches
        - (3, ): "metadata" of this item, i.e. the sequence index, image index and camera index
        '''
        img_folder, pc_folder, seq_i, img_i, cam_i = self.preprocess.dataset[index]
        img, pc, _, _ = self.preprocess.load_item(img_folder, pc_folder, img_i)
        patches, origins = self.get_patches(img)
        neighbourhoods = self.get_neighbourhoods(pc, img, seq_i, cam_i)
        return neighbourhoods, patches, origins, [seq_i, img_i, cam_i]