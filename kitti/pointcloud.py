import open3d as o3d

def compare_pc(pc1, pc2):
    def get_pc(pc, color):
        _pc = o3d.geometry.PointCloud()
        _pc.points = o3d.utility.Vector3dVector(pc)
        _pc.paint_uniform_color(color)
        return _pc
    o3d.visualization.draw_geometries([
        get_pc(pc1, [0.2, 0.2, 1]), 
        get_pc(pc2, [1, 0, 0])
    ])

def plot_pc(pc, colors=None):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    if colors is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud])