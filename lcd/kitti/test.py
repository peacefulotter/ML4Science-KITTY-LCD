import os
import numpy as np
import matplotlib.pyplot as plt

from preprocess import KittiPreprocess

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
preprocess = KittiPreprocess(root, mode='debug')
calibs = preprocess.calibs
calib = calibs[0]

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

print("pc shape ", pc.shape)
# P2 (3 x 4) for left eye
P2_rect = calib['P_rect_20']
# Add a 1 in bottom-right, reshape to 4 x 4
Tr_velo_to_cam = calib['T_cam2_velo']

# read raw data from binary
# TODO: use fov filter? 
velo = np.insert(pc.T,3,1,axis=1).T
velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
cam = P2_rect @ Tr_velo_to_cam @ velo
print("cam shape: ", cam.shape)
cam = np.delete(cam,np.where(cam[2,:]<0)[0],axis=1)
# get u,v,z
cam[:2] /= cam[2,:]
# do projection staff
plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
IMG_H,IMG_W,_ = img.shape
# restrict canvas in range
plt.axis([0,IMG_W,IMG_H,0])
plt.imshow(img)
# filter point out of canvas
u,v,z = cam
u_out = np.logical_or(u<0, u>IMG_W)
v_out = np.logical_or(v<0, v>IMG_H)
outlier = np.logical_or(u_out, v_out)
cam = np.delete(cam,np.where(outlier),axis=1)
# generate color map from depth
u,v,z = cam
plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
plt.show()