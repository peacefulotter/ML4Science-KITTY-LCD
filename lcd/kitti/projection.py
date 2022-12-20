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


def __project(calibs, pc, img, seq_i, cam_i):
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
    _, _, in_image_mask = __project(calibs, pc, img, seq_i, cam_i)
    return in_image_mask


def project(calibs, pc, img, seq_i, cam_i):

    pts_in_frame, depth_mask, _ = __project(calibs, pc, img, seq_i, cam_i)
    
    # Get RGB for each point on the image
    in_frame_mask = np.floor(pts_in_frame).astype(int)
    pixels = img[ in_frame_mask[:, 1], in_frame_mask[:, 0] ] 
    pixels = pixels / 255 # (M, 3) RGB per point
    
    # Get the pointcloud back using the masks indices
    total_mask = combine_masks(depth_mask, in_frame_mask)
    pc_in_frame = pc.T[total_mask]

    return pts_in_frame, pixels, pc_in_frame, total_mask


def project_kitti(pc, intensity, sn, img, seq_i, cam_i):

    _, pixels, pc_in_frame, mask = project(pc, img, seq_i, cam_i)

    intensity_in_frame = intensity.T[mask]
    sn_in_frame = sn.T[mask]

    return pc_in_frame, intensity_in_frame, sn_in_frame, pixels
