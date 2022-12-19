
import os
import numpy as np

path = '../poses'

def import_poses(root, seq_list):
    poses = []
    for seq_i in seq_list:
        file_path = os.path.join(root, 'poses', '%02d.txt' % seq_i )
        poses_i = np.loadtxt(file_path, delimiter=' ')
        poses_i = poses_i.reshape((poses_i.shape[0], 3, 4))
        poses.append(poses_i)
    return poses
    

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    seq_i = 0
    img_i = 0
    poses = import_poses(root, seq_i)
