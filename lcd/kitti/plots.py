import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import pointcloud 
# from pointcloud import project_image, proj_pixel_coordinates
# from .pointcloud import project_image, proj_pixel_coordinates

colors = [
    [0.2, 0.2, 1],
    [0.8, 0.2, 0.2],
    [0.2, 0.8, 0.3]
]

colors_name = [
    'blue',
    'red',
    'green'
]

def get_pc(pc, color_i):
    _pc = o3d.geometry.PointCloud()
    _pc.points = o3d.utility.Vector3dVector(pc)
    _pc.paint_uniform_color(colors[color_i])
    return _pc

def get_pc_with_color(pc, color):
    if color in colors_name:
        return get_pc(pc, colors_name.index(color))
    _pc = o3d.geometry.PointCloud()
    _pc.points = o3d.utility.Vector3dVector(pc)
    _pc.colors = o3d.utility.Vector3dVector(color)
    return _pc

def compare_pc_with_colors(*pcs):
    assert len([pc for pc in pcs if pc in colors_name or pc.shape[1] == 3]) == len(pcs)
    o3d.visualization.draw_geometries([
        get_pc_with_color(pcs[i], pcs[i + 1]) for i in range(0, len(pcs), 2)
    ])
def compare_pc(*pcs):
    assert len([pc for pc in pcs if pc.shape[1] != 3]) == 0
    o3d.visualization.draw_geometries([
        get_pc(pc, i) for i, pc in enumerate(pcs)
    ])

def plot_pc(pc, colors=None):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    if colors is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud])


def plot_projected_depth(pc, Tr, P, img, img_w, img_h):

    # Project pointcloud to the i'th image plane
    projection, depth = pointcloud.project_image(pc, Tr, P)
    pixel_coordinates, indices = pointcloud.proj_pixel_coordinates(projection, img_w, img_h)

    depth = depth[indices]
    print("pc original shape: ", pc.shape)

    pc = pc[:, indices]

    # Establish empty render image, then fill with the depths of each point
    render = np.zeros((img_h, img_w))
    for j, (u, v) in enumerate(pixel_coordinates):
        render[v, u] = depth[j]

    _, ax = plt.subplots(nrows=2)
    ax[0].imshow(img)
    ax[1].imshow(render, vmin=0, vmax=np.max(render) / 2)
    plt.show()

def plot_imgs(*imgs):
    _, axes = plt.subplots(nrows=len(imgs))
    for i, img in enumerate(imgs):
        axes[i].imshow(img)
    plt.show()