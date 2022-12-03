import numpy as np
import cv2
import constants as cst

def get_rotation_matrix(omega, phi, kappa):
    omega,phi,kappa = np.radians(omega),np.radians(phi),np.radians(kappa) #Convert to radians the angle

    rot_x=np.array([[1, 0, 0],
                    [0, np.cos(omega), -np.sin(omega)],
                    [0, np.sin(omega), np.cos(omega)]])

    rot_y=np.array([[np.cos(phi), 0, np.sin(phi)],
                    [0, 1, 0],
                    [-np.sin(phi), 0, np.cos(phi)]])

    rot_z=np.array([[np.cos(kappa), -np.sin(kappa), 0],
                    [np.sin(kappa), np.cos(kappa), 0],
                    [0, 0, 1]])

    rotation=rot_x@rot_y@rot_z

    return rotation 


'''Create intrinsic matrix according to equation given in Agisoft Metashape Manual (page 138)'''
def get_intrinsic_matrix():
    int_mat=np.zeros([3,3])
    int_mat[0,0] = -cst.IO_F + cst.IO_B1            # -f : check website pix4d linked below
    int_mat[0,1] = cst.IO_B2
    int_mat[0,2] = cst.IO_CX 
    int_mat[1,1] = -cst.IO_F
    int_mat[1,2] = cst.IO_CY 
    int_mat[2,2] = 1

    return(int_mat)    
    
   
#Rotation matrix and transition vector are defined as per description in PIX4D :
# https://support.pix4d.com/hc/en-us/articles/202559089-How-are-the-Internal-and-External-Camera-Parameters-defined
def project_points(points,x,y,z,omega,phi,kappa):
    rotation = get_rotation_matrix(omega,phi,kappa)
    rmat = np.transpose(rotation)
    tvec = -np.transpose(rotation) @ np.array([[x],[y],[z]])
    int_matrix = get_intrinsic_matrix() 
    dist_coef = np.array([cst.IO_K1, cst.IO_K2, cst.IO_P2, cst.IO_P1]) #P1 and P2 'inverted' in opencv doc
    projected_points = cv2.projectPoints(points,rmat, tvec, int_matrix,distCoeffs=dist_coef)

    return projected_points
