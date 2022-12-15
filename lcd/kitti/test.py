import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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

def read_calib_line(line):
    key, mat = extract_key_mat(line)
    return key, get_tr(mat) if key == 'Tr' else get_p(mat)

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

calib = {}
calib_path = os.path.join(root, 'calib', '00', 'calib.txt')
with open(calib_path,'r') as f:
    for line in f.readlines():
        key, mat = read_calib_line(line)
        calib[key] = mat
print(calib)


from preprocess import KittiPreprocess 
preprocess = KittiPreprocess(
    root=root,
    mode='debug',
    patch_w=64,
    patch_h=64,
    num_pc=1024,
    min_pc=32
)  

idx = 70
img_folder, pc_folder, K_folder, seq_i, img_i, key = preprocess.dataset[idx]
img, pc, intensity, sn, K = preprocess.load_item(img_folder, pc_folder, K_folder, img_i)

print(pc.shape)

# P2 (3 x 4) for left eye
P2 = calib['P2']
# R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
# Add a 1 in bottom-right, reshape to 4 x 4
# R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
# R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
# Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
# Tr_velo_to_cam =x np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
Tr = calib['Tr']

print(P2.shape, Tr.shape)

# read raw data from binary
# scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
points = pc # scan[:, 0:3] # lidar xyz (front, left, up)
print(points.shape)
velo = np.insert(points,3,1,axis=1)
velo = np.delete(velo,np.where(velo[:, 0]<0),axis=1)
# cam = P2 * R0_rect * Tr_velo_to_cam * velo
print(velo.shape)
# no other choice to add a row of one
velo = np.r_[ velo, np.ones((1,velo.shape[1])) ]

cam = (np.matmul(P2, Tr) @ velo)
print(cam.shape)
print("where", np.where(cam[2,:]<0)[0].shape)
cam = np.delete(cam,np.where(cam[2,:]<0)[0],axis=1)
print(cam.shape)
# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
png = img # mpimg.imread(img)
IMG_H,IMG_W,_ = png.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(png)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.title('Name')
# plt.savefig(f'./{"Name"}.png',bbox_inches='tight')
plt.show()