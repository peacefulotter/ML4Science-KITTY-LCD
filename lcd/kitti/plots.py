import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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

def plot_rgb_pc(rgb_pc):
    pc, colors = np.hsplit(rgb_pc, 2)
    plot_pc(pc, colors)

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

# plots.compare_pc_with_colors(
#     pc_in_frame, colors,
#     neigbors_pc, 'red',
#     np.array([center]), 'green'
# )
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

def display_points_in_image(total_mask, pc):
    colors = np.zeros(pc.shape)
    colors[1, total_mask] = 190/255 # highlight selected points in green
    colors[2, ~total_mask] = 120/255 # highlight unselected points in blue
    plot_pc(pc.T, colors.T)

def plot_img_against_pc(img, pts_in_frame, z):
    plt.figure()
    plt.imshow(img)
    plt.scatter(pts_in_frame[:, 0], pts_in_frame[:, 1], c=z, cmap='plasma_r', marker=".", s=5)
    plt.colorbar()
    plt.show()