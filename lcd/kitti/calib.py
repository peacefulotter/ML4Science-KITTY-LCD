
import os
import numpy as np

def extract_key_mat(line):
    key = line[0:2]
    mat = np.fromstring(line[4:], sep=' ')
    mat = mat.reshape((3, 4))
    mat = mat.astype(np.float32)
    return key, mat

def get_tr(mat):
    Tr = np.identity(4)
    Tr[0:3, :] = mat
    return Tr

def get_p(mat):
    K = mat[0:3, 0:3] # camera intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # (tx, ty, tz) translation vector
    tz = mat[2, 3]
    tx = (mat[0, 3] - cx * tz) / fx
    ty = (mat[1, 3] - cy * tz) / fy
    P = np.eye(N=3, M=4, k=0)
    P[:3, :3] = K
    P[0:3, 3] = np.asarray([tx, ty, tz])
    return P

def read_calib_line(calib, line):
    key, mat = extract_key_mat(line)
    calib[key] = get_tr(mat) if key == 'Tr' else get_p(mat)

def read_calib_files(path, cb):
    seq_folders = [
        name for name in os.listdir(
        os.path.join(path, 'calib'))
    ]
    for seq in seq_folders:
        file_path = os.path.join(path, 'calib', seq, 'calib.txt')
        with open(file_path, 'r') as f:
            for line in f.readlines():
                cb(seq, line)

def import_calib(path):
    calib = []
    def callback(seq, line):
        seq = int(seq)
        while len(calib) <= seq:
            calib.append({})
        read_calib_line(calib[seq], line)
    read_calib_files(path, callback)
    return calib