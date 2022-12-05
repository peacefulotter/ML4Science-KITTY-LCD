import open3d as o3d
import numpy as np

colors = [
    [0.2, 0.2, 1],
    [0.8, 0.2, 0.2],
    [0.2, 0.8, 0.3]
]

def compare_pc(*pcs):
    assert len([pc for pc in pcs if pc.shape[1] != 3]) == 0

    def get_pc(pc, color):
        _pc = o3d.geometry.PointCloud()
        _pc.points = o3d.utility.Vector3dVector(pc)
        _pc.paint_uniform_color(color)
        return _pc
    
    o3d.visualization.draw_geometries([
        get_pc(pc, colors[i]) for i, pc in enumerate(pcs)
    ])

def plot_pc(pc, colors=None):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    if colors is not None:
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pointcloud])


def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print("Reflection detected")
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return R, t


if __name__ == '__main__':
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U*Vt

    # remove reflection
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = U*Vt

    # number of points
    n = 100

    A = np.mat(np.random.rand(n,3))
    B = R*A.T + np.tile(t, (1, n))
    B = B.T

    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)

    A2 = (ret_R * A.T) + np.tile(ret_t, (1, n))
    A2 = A2.T

    # Find the error
    err = A2 - B

    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = np.sqrt(err/n);

    print("Rotation")
    print(R)
    print("")

    print("Translation")
    print(t)
    print("")

    print("RMSE:", rmse)
    print("Correct: ", np.isclose(rmse, 0))

    B = np.array(B)
    # Add small offset to be able to see the pointcloud
    A2 = np.array(A2) + np.tile([0.01, 0, 0], (n, 1))
    compare_pc(A, B, A2)
