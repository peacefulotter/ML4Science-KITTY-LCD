
import os
import numpy as np

path = '../poses'

def import_poses(root, seq_i):
    file_path = os.path.join(root, './poses', '%02d.txt' % seq_i )
    poses = np.loadtxt(file_path, delimiter=' ')
    poses = poses.reshape((poses.shape[0], 3, 4))
    print(poses.shape)
    print(poses[0])      
    return poses

def get_pos(poses, idx):
    mat = poses[idx]
    r = mat[:3, :3]
    t = mat[:, 3]
    print(r, t)
    return r, t

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    seq_i = 0
    img_i = 0
    poses = import_poses(root, seq_i)
    get_pos(poses, img_i)