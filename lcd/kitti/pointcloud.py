import numpy as np


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


def downsample(arr, num):
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


def points_in_radius(pc, radius=1, num=1024):
    print('Computing KDTree query_ball_point for', pc.shape[0], 'points')
    import scipy.spatial as spatial
    tree = spatial.KDTree(pc)
    ball_points = tree.query_ball_point(pc, r=radius)
    neighbors = np.zeros((pc.shape[0], num))
    for i, indices in enumerate(ball_points):
        indices = np.array(indices)
        downsample_indices = downsample(indices, num=num)
        neighbors[i] = downsample_indices
    return neighbors



def project_image(pc, Tr, P):
    # We know the lidar X axis points forward, we need nothing behind the lidar, so we
    # ignore anything with a X value less than or equal to zero
    pc = pc[pc[:, 0] > 0].T
    
    # Add row of ones to make coordinates homogeneous for tranformation into the camera coordinate frame
    pc = np.hstack([pc, np.ones(pc.shape[0]).reshape((-1,1))])

    # Transform pointcloud into camera coordinate frame
    cam_xyz = Tr.dot(pc.T)
    
    # Ignore any points behind the camera (probably redundant but just in case)
    cam_xyz = cam_xyz[:, cam_xyz[2] > 0]
    
    # Extract the Z row which is the depth from camera
    depth = cam_xyz[2].copy()
    
    # Project coordinates in camera frame to flat plane at Z=1 by dividing by Z
    cam_xyz /= cam_xyz[2]
    
    # Add row of ones to make our 3D coordinates on plane homogeneous for dotting with P0
    # cam_xyz = np.vstack([cam_xyz, np.ones(cam_xyz.shape[1])])

    # Get pixel coordinates of X, Y, Z points in camera coordinate frame
    projection = P.dot(cam_xyz)
    # projection = projection / projection[2]

    return projection # , depth

def proj_pixel_coordinates(projection, img_w, img_h):
    # Turn pixels into integers for indexing
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
    # Limit pixel coordinates considered to those that fit on the image plane
    indices = np.where(
        (pixel_coordinates[:, 0] < img_w) & 
        (pixel_coordinates[:, 0] >= 0) & 
        (pixel_coordinates[:, 1] < img_h) & 
        (pixel_coordinates[:, 1] >= 0)
    )
    pixel_coordinates = pixel_coordinates[indices]
    return pixel_coordinates, indices[0]

def pointcloud2image(pc, Tr, P, img_w, img_h):
    '''
    Takes a pointcloud of shape Nx4 and projects it onto an image plane, first transforming
    the X, Y, Z coordinates of points to the camera frame with tranformation matrix Tr, then
    projecting them using camera projection matrix P0.
    
    Arguments:
    pointcloud -- array of shape Nx4 containing (X, Y, Z, reflectivity)
    imheight -- height (in pixels) of image plane
    imwidth -- width (in pixels) of image plane
    Tr -- 3x4 transformation matrix between lidar (X, Y, Z, 1) homogeneous and camera (X, Y, Z)
    P0 -- projection matrix of camera (should have identity transformation if Tr used)
    
    Returns:
    render -- a (imheight x imwidth) array containing depth (Z) information from lidar scan
    
    '''
    # Project pointcloud to the i'th image plane
    projection, _ = project_image(pc, Tr, P)
    _, indices = proj_pixel_coordinates(projection, img_w, img_h)
    pc = pc[:, indices]
    return pc


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
    neighbors_indices = points_in_radius(pc)
    print(neighbors_indices.shape)

    for i, indices in enumerate(neighbors_indices):
        center = pc[i]
        neighbors = pc[indices]
        compare_pc( pc, neighbors, np.array([center]) )
