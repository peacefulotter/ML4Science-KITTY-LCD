import numpy as np

from loguru import logger

def downsample_arr(arr, num):
    nb_points = arr.shape[0]
    if nb_points >= num:
        choice_idx = np.random.choice(nb_points, num, replace=False)
    else:
        fix_idx = np.asarray(range(nb_points))
        while nb_points + fix_idx.shape[0] < num:
            fix_idx = np.concatenate((fix_idx, np.asarray(range(nb_points))), axis=0)
        random_idx = np.random.choice(nb_points, num - fix_idx.shape[0], replace=False)
        choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
    return arr[choice_idx]


"""
Uses all the points for finding neighbors
but compute the query_ball_point only for the voxel downsampled points
this reduces computation (on less points) while keeping
the "important" points 
"""
def downsample_neighbors(ds_pc, pc, min_neighbors, radius=1, downsample=1024):
    '''
    ds_pc: voxel downsampled pointcloud
    pc: poincloud that projected onto the image
    min_neighbors: minimum amount of neighbors to keep the point
    radius: radius of points to return, in meters
    downsample: downsamples the neighbors to be this amount (duplicate)
    '''
    logger.info(f'Computing KDTree query_ball_point for {ds_pc.shape[0]} points with {pc.shape[0]} total points')
    import scipy.spatial as spatial
    tree = spatial.KDTree(pc)
    ball_points = tree.query_ball_point(ds_pc, r=radius)
    neighbors = np.zeros((pc.shape[0], downsample), dtype=int)
    centers = np.zeros((pc.shape[0], 3))
    count = 0
    for i, indices in enumerate(ball_points):
        indices = np.array(indices)
        if indices.shape[0] < min_neighbors:
            continue
        downsample_indices = downsample_arr(indices, num=downsample)
        neighbors[count] = downsample_indices
        centers[count] = ds_pc[i]
        count += 1
    logger.info(f'Found neighbors for {count} points')
    return neighbors[:count, :], centers[:count]

# def project_image(pc, Tr, P):
#     # We know the lidar X axis points forward, we need nothing behind the lidar, so we
#     # ignore anything with a X value less than or equal to zero
#     pc = pc[pc[:, 0] > 0].T
    
#     # Add row of ones to make coordinates homogeneous for tranformation into the camera coordinate frame
#     pc = np.hstack([pc, np.ones(pc.shape[0]).reshape((-1,1))])

#     # Transform pointcloud into camera coordinate frame
#     cam_xyz = Tr.dot(pc.T)
    
#     # Ignore any points behind the camera (probably redundant but just in case)
#     cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
    
#     # Extract the Z row which is the depth from camera
#     depth = cam_xyz[2].copy()
    
#     # Project coordinates in camera frame to flat plane at Z=1 by dividing by Z
#     cam_xyz /= cam_xyz[2]
    
#     # Add row of ones to make our 3D coordinates on plane homogeneous for dotting with P0
#     # cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])

#     # Get pixel coordinates of X, Y, Z points in camera coordinate frame
#     projection = P.dot(cam_xyz)
#     # projection = projection / projection[2]

#     return projection # , depth


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


def test_rigid_transform():
    R = np.mat(np.random.rand(3,3))
    t = np.mat(np.random.rand(3,1))

    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U * Vt

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
    rmse = np.sqrt(err/n)

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
    
    from .plots import compare_pc 
    compare_pc(A, B, A2)

if __name__ == '__main__':
    # test_rigid_transform()
    from dataset import KittiDataset 
    from plots import compare_pc

    dataset = KittiDataset(
        root="../../", 
        mode='train', 
        num_pc=4096, 
        img_width=256, 
        img_height=256
    )
    pc, img = dataset[0]
    pc, colors = np.hsplit(pc, 2)
    neighbors_indices = downsample_neighbors(pc)
    print(neighbors_indices.shape)

    for i, indices in enumerate(neighbors_indices):
        center = pc[i]
        neighbors = pc[indices]
        compare_pc( pc, neighbors, np.array([center]) )
