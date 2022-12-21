import numpy as np

def combine_masks(depth_mask, in_frame_mask):
    '''
    Combine sequential masks where the second mask is used after the first one
    '''
    mask = np.zeros(depth_mask.shape)
    idx_in_frame = 0
    for idx_depth, depth in enumerate(depth_mask):
        if depth:
            mask[idx_depth] = in_frame_mask[idx_in_frame]
            idx_in_frame += 1
    return mask.astype(bool)


def __project(calibs, pc, seq_i, cam_i):
    '''
    The actual projection function: 
    1. Turns the pointcloud coordinates into homogeneous coordinates (x, y, z, 1)
    2. Apply multiply the homogeneous coordinates with the  projection matrix 
    after rectification Pi (for the ith camera) and the Ti matrix that transforms 
    the points from the velodyne coordinates into the left rectified camera coordinate
    system. 
    3. Extract the depth channel and discard the projected points with a depth < 0.1
    4. Divide the points with the z coordinates to get the (u, v) coordinates which
    are the pixel coordinates in the img.

    Params:
        - calibs: Map of key -> matrices imported and parsed from a kitti calib file
        and that contains at least Pi and Ti for the ith camera the projection is onto
        - (3, N) pc: Pointcloud 
        - seq_i: The ith sequence
        - cam_i: The ith camera
    
    Returns:
        - (n, 3) pts_on_camera_place: 2D projected points in the camera plane 
        - (N, ) depth_mask: Mask used to filter the points that are projected behind the camera

    '''
    # Get Pi and Ti from the calib file
    calib = calibs[seq_i]
    Pi = calib[f'P{cam_i}']
    Ti = calib[f'T{cam_i}'] # {cam_i}

    # Project onto the image
    pc_ext = np.r_[ pc, np.ones((1,pc.shape[1])) ]
    pts_cam = (Pi @ Ti @ pc_ext)[:3, :].T

    # discard the points behind the camera (of negative depth) -- these points get flipped during the z_projection
    depth = pts_cam[:, 2]
    depth_mask = ~(depth < 0.1)
    pts_front_cam = pts_cam[depth_mask]

    z = pts_front_cam[:, 2:3]
    pts_on_camera_plane = pts_front_cam / z

    return pts_on_camera_plane, depth_mask

def __project_to_img(calibs, pc, img, seq_i, cam_i):
    '''
    Projects the pointcloud (pc) into the image frame 
    and cuts the points that are projected outside the image bounds.
    
    Params:
        - calibs: Map of key -> matrices imported and parsed from a kitti calib file
        and that contains at least Pi and Ti for the ith camera the projection is onto
        - (3, N) pc: Pointcloud 
        - (H, W, 3) img: RGB image where each channel is in the range [0, 255]
        - seq_i: The ith sequence
        - cam_i: The ith camera

    Returns:
        - (n, 3) pts_in_frame: 2D projected points in the camera frame 
        - (N, ) depth_mask: Mask used to filter the points that are projected behind the camera
        - (N, ) in_image_mask: Mask used to filter the points that are projected outside the img bounds
    '''
    pts_on_camera_plane, depth_mask = __project(calibs, pc, seq_i, cam_i)
    pts_on_camera_plane[:, 1] = pts_on_camera_plane[:, 1] - img.shape[0] / 6

    # take the points falling inside the image
    in_image_mask = (
        (pts_on_camera_plane[:, 0] >= 0) &
        (pts_on_camera_plane[:, 0] < img.shape[1]) &
        (pts_on_camera_plane[:, 1] >= 0) &
        (pts_on_camera_plane[:, 1] < img.shape[0])
    )
    pts_in_frame = pts_on_camera_plane[in_image_mask]
    return pts_in_frame, depth_mask, in_image_mask


def projected_img_indices(calibs, pc, img, seq_i, cam_i):
    '''
    Retrieves the mask used to filter the points projected outside the image bounds. 
    This function projects the pointcloud (pc) into the image frame (img)
    using the matrices in the calibs file (calibs) and discards the points that 
    are projected behind the camera frame and/or outside the image bounds. 

    Params:
        - calibs: Map of key -> matrices imported and parsed from a kitti calib file
        and that contains at least Pi and Ti for the ith camera the projection is onto
        - (3, N) pc: Pointcloud 
        - (H, W, 3) img: RGB image where each channel is in the range [0, 255]
        - seq_i: The ith sequence
        - cam_i: The ith camera

    Returns:
        - (N, ) in_image_mask: Mask used to filter the points that are projected outside the img bounds
    '''
    _, _, in_image_mask = __project_to_img(calibs, pc, img, seq_i, cam_i)
    return in_image_mask


def project(calibs, pc, img, seq_i, cam_i):
    '''
    Projects the pointcloud (pc) into the image frame (img)
    using the matrices in the calibs file (calibs).

    Moreover, this function gets the 2d projected points, transforms
    them into img coordinates (by flooring the values) and retrieves 
    the pixel color for each coordinate. These colors correspond to the 
    RGB value of each projected points in the pointcloud.

    Finally, retrieve the 3D points that can be projected onto the image using
    a combination of the "depth_mask" and "in_img_mask" 

    Params:
        - calibs: Map of key -> matrices imported and parsed from a kitti calib file
        and that contains at least Pi and Ti for the ith camera the projection is onto
        - (3, N) pc: Pointcloud 
        - (H, W, 3) img: RGB image where each channel is in the range [0, 255]
        - seq_i: The ith sequence
        - cam_i: The ith camera

    Returns:
        - (n, 3) pts_in_frame: 2D projected points in the camera frame 
        - (n, 3) pixels: RGB color values for each projected points
        - (n, 3) pc_in_frame: 3D points from the pointcloud that can be projected onto the image bounds
        - (N, ) total_mask: mask used to retrieve the pc_in_frame from the original pointcloud (pc)
    '''
    pts_in_frame, depth_mask, in_img_mask = __project_to_img(calibs, pc, img, seq_i, cam_i)
    
    # Get RGB for each point on the image
    in_frame_mask = np.floor(pts_in_frame).astype(int)
    pixels = img[ in_frame_mask[:, 1], in_frame_mask[:, 0] ] 
    pixels = pixels / 255 # (M, 3) RGB per point
    
    # Get the pointcloud back using the masks indices
    total_mask = combine_masks(depth_mask, in_img_mask)
    pc_in_frame = pc.T[total_mask]

    return pts_in_frame, pixels, pc_in_frame, total_mask


def project_kitti(calibs, pc, intensity, sn, img, seq_i, cam_i):
    '''
    This function is exactly projection.project but it also retrieves the 
    intensity and surface normals (sn) that correspond to the projected 3D points.
    Basically, if from the pointcloud with retrieve the ones that can be projected in 
    the image bounds using a mask "mask", then it also applies this mask to the given
    intensity and sn arrays.

    Params:
        - calibs: Map of key -> matrices imported and parsed from a kitti calib file
        and that contains at least Pi and Ti for the ith camera the projection is onto
        - (3, N) pc: Pointcloud
        - (1, N) intensity: Intensity array from the same pointcloud
        - (3, N) sn: Surface normals array from the same pointcloud
        - (H, W, 3) img: RGB image where each channel is in the range [0, 255]
        - seq_i: The ith sequence
        - cam_i: The ith camera

    Returns:
        - (n, 3) pc_in_frame: 3D points from the pointcloud that can be projected onto the image bounds
        - (n, 1) intensity_in_frame: Intensity array after applying the same mask as the pointcloud
        - (n, 3) sn_in_frame: Surface normals array after applying the same mask as the pointcloud
        - (n, 3) pixels: RGB color values for each projected points
    '''
    _, pixels, pc_in_frame, mask = project(calibs, pc, img, seq_i, cam_i)

    print(intensity.shape, sn.shape)

    intensity_in_frame = intensity.T[mask]
    sn_in_frame = sn.T[mask]

    return pc_in_frame, intensity_in_frame, sn_in_frame, pixels
