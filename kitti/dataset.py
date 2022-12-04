
import os
import torch
import numpy as np
import torch.utils.data as data
from matplotlib import pyplot as plt

from calib import import_calib
from pointcloud import plot_pc, compare_pc

# TODO: ImageFolder dataset
class KittyDataset(data.Dataset):
    def __init__(self, root, mode, num_pc, img_width, img_height, *args, **kwargs):
        super(KittyDataset, self).__init__(*args, **kwargs)
        self.root = root
        self.mode = mode
        self.num_pc = num_pc
        self.img_width = img_width
        self.img_height = img_height
        self.dataset = self.make_kitti_dataset()
        self.calib = import_calib(root)

    def get_dataset_folder(self, seq, name):
        return os.path.join(self.root, 'sequences', '%02d' % seq, name)

    def make_kitti_dataset(self):
        dataset = []

        # TODO: replace with list(range(9)) 
        seq_list = [0] if self.mode == 'train' else [9, 10]

        skip_start_end = 0
        for seq in seq_list:
            # TODO: use non-preprocessed img? 
            img2_folder = self.get_dataset_folder(seq, 'img_P2')
            img3_folder = self.get_dataset_folder(seq, 'img_P3')
            pc_folder = self.get_dataset_folder(seq, 'pc_npy_with_normal')

            sample_num = round(len(os.listdir(img2_folder)))

            for i in range(skip_start_end, sample_num - skip_start_end):
                dataset.append((img2_folder, pc_folder,
                                seq, i, 'P2', sample_num))
                dataset.append((img3_folder, pc_folder,
                                seq, i, 'P3', sample_num))
        return dataset

    def load_npy(self, folder, seq_i):
        return np.load(os.path.join(folder, '%06d.npy' % seq_i))

    def load_item(self, img_folder, pc_folder, seq_i):
        img = self.load_npy(img_folder, seq_i)
        data = self.load_npy(pc_folder, seq_i)
        pc = data[:3, :]
        return img, pc

    def __len__(self):
        return len(self.dataset) 



    def pointcloud2image(self, pc, Tr, P):
        '''
        Takes a pointcloud of shape Nx4 and projects it onto an image plane, first transforming
        the X, Y, Z coordinates of points to the camera frame with tranformation matrix Tr, then
        projecting them using camera projection matrix P0.
        
        Arguments:
        pointcloud -- array of shape Nx4 containing (X, Y, Z, reflectivity)
        imheight -- height (in pixels) of image plane
        imwidth -- width (in pixels) of image plane
        Tr -- 3x4 transformation matrix between lidar (X, Y, Z, 1) homogeneous and camera (X, Y, Z)
        P0 -- projection matrix of camera (should have identity transformation if Tr used)
        
        Returns:
        render -- a (imheight x imwidth) array containing depth (Z) information from lidar scan
        
        '''
        # We know the lidar X axis points forward, we need nothing behind the lidar, so we
        # ignore anything with a X value less than or equal to zero
        pc = pc[pc[:, 0] > 0].T
        
        # Add row of ones to make coordinates homogeneous for tranformation into the camera coordinate frame
        pc = np.hstack([pc, np.ones(pc.shape[0]).reshape((-1,1))])

        # Transform pointcloud into camera coordinate frame
        cam_xyz = Tr.dot(pc.T)
        
        # Ignore any points behind the camera (probably redundant but just in case)
        cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
        
        # Extract the Z row which is the depth from camera
        depth = cam_xyz[2].copy()
        
        # Project coordinates in camera frame to flat plane at Z=1 by dividing by Z
        cam_xyz /= cam_xyz[2]
        
        # Add row of ones to make our 3D coordinates on plane homogeneous for dotting with P0
        cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])
        
        # Get pixel coordinates of X, Y, Z points in camera coordinate frame
        projection = P.dot(cam_xyz)
        # projection = (projection / projection[2])

        # Turn pixels into integers for indexing
        pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
        #pixel_coordinates = np.array(pixel_coordinates)
        
        # Limit pixel coordinates considered to those that fit on the image plane
        indices = np.where((pixel_coordinates[:, 0] < self.img_width)
                        & (pixel_coordinates[:, 0] >= 0)
                        & (pixel_coordinates[:, 1] < self.img_height)
                        & (pixel_coordinates[:, 1] >= 0)
                        )
        pixel_coordinates = pixel_coordinates[indices]
        depth = depth[indices]
        pc = pc[indices]
        
        # Establish empty render image, then fill with the depths of each point
        render = np.zeros((self.img_height, self.img_width))
        for j, (u, v) in enumerate(pixel_coordinates):
            render[v, u] = depth[j]

        return pc[:, :3], render

    def __getitem__(self, index):
        img_folder, pc_folder, seq, seq_i, key, _ = self.dataset[index]
        img, pc = self.load_item(img_folder, pc_folder, seq_i)

        # Project point cloud to image
        Pi = self.calib[seq][key] # (3, 4)
        Tr = self.calib[seq]['Tr'] # (4, 4)

        in_img_pc, render = self.pointcloud2image(pc, Tr, Pi)
        print(render.shape)
        fig, ax = plt.subplots(nrows=2)
        ax[0].imshow(img)
        ax[1].imshow(render, vmin=0, vmax=np.max(render) / 2)
        plt.show()
        print(pc.shape)
        compare_pc(pc.T, in_img_pc)
        
        return 0, 0

        X = np.r_[ pc, np.ones((1, pc.shape[1])) ]   # (X, Y, Z, 1), shape=(4, N)
        x = Pi @ Tr @ X # (3, N) where the last row is not used
        # print(x.T)
        x = np.delete(x, (0), axis=0) # (2, N)

        
        # plot_pc(render)

        # TODO: take random subpart of img
        # of size (img_width, img_height)

        # Mask the pc to only keep the points projected onto the image
        in_img_mask = self.bounding_box(x.T, 0, self.img_width, 0, self.img_height) # (M, )
        print("Keeping nb points =", in_img_mask[in_img_mask == True].shape[0])
        x_in_img = x[:, in_img_mask] # (2, M)
        pc_in_img = pc[:, in_img_mask] # (3, M)
        
        # get RGB color of each point in the pc_in_img
        pc_img_mask = np.round(x_in_img).astype(int) # (2, M) (int)
        print(x_in_img.T)
        colors = img[ pc_img_mask[0, :], pc_img_mask[1, :] ] # (M, 3) RGB per point

        """plt.imshow(colors.T) #, interpolation='nearest'
        plt.show()"""

        # concatenate pc with rgb color
        pc = np.r_[ pc_in_img, colors.T / 255 ]  
        return img, pc


if __name__ == '__main__':
    # idx = 50
    w, h = 1226, 360
    dataset = KittyDataset(root="./", mode='train', num_pc=0, img_width=w, img_height=h)
    for idx in range(0, 200, 2):
        img, pc_colored = dataset[idx]
        # pc, colors = np.vsplit(pc_colored, 2)
        # plot_pc(pc, colors)