
import os
import numpy as np

path = '../poses'

def import_poses(seq_i):
    file_path = os.path.join('./poses', '%02d.txt' % seq_i )
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
    poses = import_poses(0)
    get_pos(poses, 0)