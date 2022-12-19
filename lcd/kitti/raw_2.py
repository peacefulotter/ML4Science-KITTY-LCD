
import os
import cv2
import numpy as np
from plots import plot_pc
import matplotlib.pyplot as plt

from PIL import Image

from calib import read_calib_file, parse_calib_file, get_p

if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')

    img = np.load(os.path.join(root, "sequences", "00", "img_P2", '%06d.npy' % 0))
    data = np.load(os.path.join(root, "sequences", "00", "pc_npy_with_normal", '%06d.npy' % 0))
    pc = data[0:3, :]
    img_h, img_w, _ = img.shape

    calib = read_calib_file(root, 0)
    calib = parse_calib_file(calib)

    Pi = calib['P2']
    Tr = calib['Tr']
    K = Pi[:3, :3]
    # Pi = get_p(Pi)

    Tr_hg = np.identity(4)
    Tr_hg[:3, :] = Tr

    print(Pi.shape, Tr.shape)

    pc_hg = np.r_[ pc, np.ones((1,pc.shape[1])) ]

    pts_2d_hg = (Pi @ Tr_hg) @ pc_hg
    print(pts_2d_hg.shape)
    z = pts_2d_hg[2]
    pts_2d = pts_2d_hg[:2] / z
    print(pts_2d.shape)
    print(pts_2d)

    # pts_2d = []
    # for i in range(pc.shape[1]):
    #     pt = pc[:, i]
    #     pt_hg = np.append(pt, 1)
    #     point_c = Rt @ pt_hg
    #     xc, yc, zc = point_c
    #     fx = Pi[0, 0]
    #     fy = Pi[1, 1]
    #     cx = Pi[0, 2]
    #     cy = Pi[1, 2]
    #     u = fx * xc / zc + cx
    #     v = fy * yc / zc + cy
    #     s = 100
    #     pts_2d.append([u / s, v / s, zc])

    # take the points falling inside the image
    pixels = np.floor(pts_2d).astype(int)

    print(pixels.shape)

    u, v = pixels
    in_image_mask = (
        (u >= 0) & (u < img_w) &
        (v >= 0) & (v < img_h)
    )
    pixels_in_frame = pixels[:, in_image_mask]
    z = z[in_image_mask]

    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    u, v = pixels_in_frame
    plt.scatter(u, v, c=z, cmap='plasma_r', marker=".", s=5)
    plt.colorbar()
    plt.show()