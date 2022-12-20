
import os
import numpy as np

# parse line and extract key and corresponding matrix
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


def transform_calib(filedata):
    data = {}

    # data['Tr'] = filedata['Tr']

    P_rect_00 = np.reshape(filedata['P0'], (3, 4))
    P_rect_10 = np.reshape(filedata['P1'], (3, 4))
    P_rect_20 = np.reshape(filedata['P2'], (3, 4))
    P_rect_30 = np.reshape(filedata['P3'], (3, 4))

    # P_rect_00
    data['P0'] = P_rect_00
    data['P1'] = P_rect_10
    data['P2'] = P_rect_20
    data['P3'] = P_rect_30

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    data['T0'] = np.reshape(filedata['Tr'], (3, 4))
    data['T0'] = np.vstack([data['T0'], [0, 0, 0, 1]])
    data['T1'] = T1.dot(data['T0'])
    data['T2'] = T2.dot(data['T0'])
    data['T3'] = T3.dot(data['T0'])

    # Compute the camera intrinsics
    data['K0'] = P_rect_00[0:3, 0:3]
    data['K1'] = P_rect_10[0:3, 0:3]
    data['K2'] = P_rect_20[0:3, 0:3]
    data['K3'] = P_rect_30[0:3, 0:3]

    return data

def read_calib_file(root, seq_i):
    # file_path = os.path.join(root, 'calib', '%02d' % seq_i, 'calib.txt')
    # file_path = os.path.join(root, 'sequences', '%02d' % seq_i, 'calib_corr.txt')
    file_path = os.path.join(root, 'sequences', '%02d' % seq_i, 'calib.txt')
    with open(file_path, 'r') as f:
        return f.readlines()

def parse_calib_file(filedata):
    res = {}
    for line in filedata:
        key, mat = extract_key_mat(line)
        res[key] = mat
    return res


def import_calibs(root, sequences):
    print(f' > Importing calib file for {len(sequences)} sequences')
    calibs = []
    for seq_i in sequences:
        data = read_calib_file(root, seq_i)
        data = parse_calib_file(data)
        data = transform_calib(data)
        calibs.append(data)
    return calibs