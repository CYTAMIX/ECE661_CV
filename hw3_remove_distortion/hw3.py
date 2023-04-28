# ECE661 HW3
# Zhengxin Jiang
# jiang839

import cv2 as cv
import numpy as np
import matplotlib . pyplot as plt


### function definetion ###



# The function takes two arrays of points and return the homography
def findHomograpyMatrix_P2P(points_1, points_2):
    
    # create a linear system to solve H
    P = np.zeros((9,9))
    
    for i in range(4):
        
        P[2*i, 0] = -points_1[i][0]
        P[2*i, 1] = -points_1[i][1]
        P[2*i, 2] = -1
        P[2*i, 6] = points_1[i][0]*points_2[i][0]
        P[2*i, 7] = points_1[i][1]*points_2[i][0]
        P[2*i, 8] = points_2[i][0]
        
        P[2*i+1, 3] = -points_1[i][0]
        P[2*i+1, 4] = -points_1[i][1]
        P[2*i+1, 5] = -1
        P[2*i+1, 6] = points_1[i][0]*points_2[i][1]
        P[2*i+1, 7] = points_1[i][1]*points_2[i][1]
        P[2*i+1, 8] = points_2[i][1]
    
    P[8, 8] = 1
    
    # take the last col of P inverse
    H = np.linalg.inv(P)[:, 8]
    H = np.reshape(H, (3,3))
    
    return H


# The function finds the homography that creates projective distortion based on the given point set
def findHomograpyMatrix_Projective(points):
    
    l1, l2 = np.cross(points[0], points[3]), np.cross(points[1], points[2])
    l3, l4 = np.cross(points[0], points[1]), np.cross(points[3], points[2])    
    p, q = np.cross(l1, l2), np.cross(l3, l4)
    p, q = p/p[2], q/q[2]
    vl = np.cross(p, q)
    vl = vl/vl[2]
    
    H = np.identity(3)
    H[2] = vl
    
    # H maps vl to vl_inf, we want the inverse H
    return np.linalg.inv(H)


# The function finds the homography that creates affine distortion based on the given point set
def findHomograpyMatrix_Affine(points):
    
    l1 = np.cross(points.T[0], points.T[1])
    l2 = np.cross(points.T[0], points.T[3])
    l3 = np.cross(points.T[1], points.T[2])
    l1 = l1/l1[2]
    l2 = l2/l2[2]
    l3 = l3/l3[2]

    # create a linear system to solve S
    A_ = np.zeros((2,2))
    A_[0] = [l1[0]*l2[0], l1[0]*l2[1]+l1[1]*l2[0]]
    A_[1] = [l1[0]*l3[0], l1[0]*l3[1]+l1[1]*l3[0]]

    b = [-l1[1]*l2[1], -l1[1]*l3[1]]

    S_ = np.linalg.solve(A_,b)
    
    # reshape S into 2x2 matrix
    S = np.ones((2,2))
    S[0][0] = S_[0]
    S[0][1] = S_[1]
    S[1][0] = S_[1]
    
    # solve A from S
    u , d, v = np.linalg.svd(S)     
    d = np.sqrt(d)     
    lamda = np.diag(d)     
    A = np.dot(np.dot(u,lamda),np.transpose(u)) 

    # turn A into the 3x3 matrix H
    H_affine = np.zeros((3,3))
    H_affine[:2, :2] = A
    H_affine[2][2] = 1

    return H_affine


# The function finds the homography using one step method based on the given point set
def findHomograpyMatrix_Onestep(points, points2):
    
    l1 = np.cross(points[0], points[1])
    l2 = np.cross(points[1], points[2])
    l3 = np.cross(points[2], points[3])
    l4 = np.cross(points[3], points[0])
    l5 = np.cross(points2[0], points2[2])
    l6 = np.cross(points2[1], points2[3])

    # 5 orthogonal line pairs
    ls = [(l1,l2), (l2,l3), (l3,l4), (l4,l1), (l5,l6)]
    
    A_ = np.zeros((5,5))
    b = np.zeros(5)
    
    # create a linear system Ax = b to solve the degenerate conic
    for i in range(5):
        A_[i] = [ls[i][0][0]*ls[i][1][0], (ls[i][0][0]*ls[i][1][1]+ls[i][0][1]*ls[i][1][0])/2, 
                 ls[i][0][1]*ls[i][1][1], (ls[i][0][0]*ls[i][1][2]+ls[i][0][2]*ls[i][1][0])/2, 
                 (ls[i][0][1]*ls[i][1][2]+ls[i][0][2]*ls[i][1][1])/2]
        b[i] = -ls[i][0][2]*ls[i][1][2]
        
    C_ = np.linalg.solve(A_,b)
    C_ = C_/np.max(C_) 
    
    # Solve A and V
    S = np.zeros((2,2))
    S[0][0] = C_[0]
    S[0][1] = C_[1]/2
    S[1][0] = C_[1]/2
    S[1][1] = C_[2]

    u , d_square, v = np.linalg.svd(S)     
    d = np.sqrt(d_square)     
    D = np.diag(d)     
    A = np.dot(np.dot(u,D),np.transpose(u)) 
    
    b2 = [C_[3]/2, C_[4]/2]
    V = np.linalg.solve(A,b2)

    H = np.zeros((3,3))
    H[:2, :2] = A
    H[2, :2] = V
    H[2][2] = 1
    
    return H


# The function creates an blank image with the same size as the input image 
def getBlankImage(width, height):
    
    blankimg = np.zeros((min(height, 50000), min(width, 50000), 3), dtype=np.uint8)
    
    return blankimg

# recover an image using a given homography
def imageRecovery(img, H):
    
    # use the origin image to calculate the size of the recovered image
    maxcoord_distort = np.array([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    maxcoord_distort = np.append(maxcoord_distort, np.ones((4,1)), axis=1)
    
    maxcoord_world = np.linalg.inv(H).dot(maxcoord_distort.T)
    maxcoord_world = (maxcoord_world/maxcoord_world[-1]).astype(int)
    
    # offset in the recovered image coordinates
    offset_x = min(maxcoord_world[0])
    offset_y = min(maxcoord_world[1])

    # calculated size of the recovered image
    new_width = max(maxcoord_world[0]) - min(maxcoord_world[0])
    new_height = max(maxcoord_world[1]) - min(maxcoord_world[1])
    
    new_img = getBlankImage(new_width, new_height)
    
    # pixel replacement
    for i in range(new_img.shape[1]):
        for j in range(new_img.shape[0]):

            x = i + offset_x
            y = j + offset_y

            proj_coord = H.dot([x, y, 1])
            x_proj = round(proj_coord[0]/proj_coord[2])
            y_proj = round(proj_coord[1]/proj_coord[2])

            # replace the projected pixel
            if 0 <= x_proj and x_proj < img.shape[1] and 0 <= y_proj and y_proj < img.shape[0]:
                new_img[j, i] = img[y_proj, x_proj]
    
#     return new_img
    return new_img


# The function recovery a distort image back into world coordinate
# using the point to point method
def pointToPointRecovery(img, coord_distort, coord_world):
    
    H = findHomograpyMatrix_P2P(coord_world, coord_distort)
    
    new_img = imageRecovery(img, H)
    
    return new_img


# The function recovery a distort image back into world coordinate
# using the two step method
def twoStepRecovery(img, coord_distort):
    
    
    coord_distort = np.append(coord_distort, np.ones((4,1)), axis=1)
    
    # remove projection distortion
    H_projective = findHomograpyMatrix_Projective(coord_distort)
    
    new_img_step1 = imageRecovery(img, H_projective)
    
    # remove affine distortion
    coord_distort_no_projective = np.linalg.inv(H_projective).dot(coord_distort.T)
    coord_distort_no_projective = coord_distort_no_projective/coord_distort_no_projective[2]
    
    H_affine = findHomograpyMatrix_Affine(coord_distort_no_projective)
#     print(H_affine)

    new_img_step2 = imageRecovery(new_img_step1, H_affine)

    return new_img_step1, new_img_step2


# The function recovery a distort image back into world coordinate
# using the two step method
def oneStepRecovery(img, coord_distort, coord_distort2):
    
    coord_distort = np.append(coord_distort, np.ones((4,1)), axis=1)
    coord_distort2 = np.append(coord_distort2, np.ones((4,1)), axis=1)
    
    H = findHomograpyMatrix_Onestep(coord_distort, coord_distort2)
    
    new_img = imageRecovery(img, H)
    
    return new_img
    
    
if __name__ == '__main__' :
    
    # images for task 1
    img_a = cv.imread('hw3images/building.jpg')
    img_b = cv.imread('hw3images/nighthawks.jpg')
    img_a=cv.cvtColor(img_a,cv.COLOR_BGR2RGB)
    img_b=cv.cvtColor(img_b,cv.COLOR_BGR2RGB)
    
    corners_a = np.array([[240, 200], [236, 369], [295, 374], [297, 215]])
    corners_b = np.array([[75, 179], [78, 652], [805, 620], [803, 219]])
    corners_a_world = np.array([[0, 0], [0, 90], [30, 90], [30, 0]])
    corners_b_world = np.array([[0, 0], [0, 85], [150, 85], [150, 0]])
    
    
    ### Task 1.1 ###
    img_a_1_1 = pointToPointRecovery(img_a, corners_a , corners_a_world)
    img_b_1_1 = pointToPointRecovery(img_b, corners_b , corners_b_world)
    
    plt.imshow(img_a_1_1)
    plt.figure()
    plt.imshow(img_b_1_1)
    plt.figure()
    
    
    ### Task 1.2 ###
    img_a_1_2_proj, img_a_1_2_aff = twoStepRecovery(img_a, corners_a)
    img_b_1_2_proj, img_b_1_2_aff = twoStepRecovery(img_b, corners_b)

    plt.imshow(img_a_1_2_proj)
    plt.figure()
    plt.imshow(img_a_1_2_aff)
    plt.figure()
    plt.imshow(img_b_1_2_proj)
    plt.figure()
    plt.imshow(img_b_1_2_aff)
    plt.figure()
    
    
    ### Task 1.3 ###
    
    # one more set of points for the 5th orthogonal line pair
    corners_a2 = np.array([[240, 200], [238, 262], [296, 271], [297, 215]])
    corners_b2 = np.array([[75, 179], [78, 652], [555, 632], [552, 204]])
    
    img_a_1_3 = oneStepRecovery(img_a, corners_a , corners_a2)
    img_b_1_3 = oneStepRecovery(img_b, corners_b , corners_b2)

    plt.imshow(img_a_1_3)
    plt.figure()
    plt.imshow(img_b_1_3)
    plt.figure()


    # images for task 2
    img_a = cv.imread('hw3images/monitor.jpg')
    img_b = cv.imread('hw3images/sticker.jpg')
    img_a=cv.cvtColor(img_a,cv.COLOR_BGR2RGB)
    img_b=cv.cvtColor(img_b,cv.COLOR_BGR2RGB)
    
    corners_a = np.array([[464, 284], [186, 895], [1470, 979], [1630, 180]])
    corners_b = np.array([[56, 78], [61, 1505], [694, 1861], [848, 360]])
    corners_a_world = np.array([[0, 0], [0, 1080], [1920, 1080], [1920, 0]])
    corners_b_world = np.array([[0, 0], [0, 900], [400, 900], [400, 0]])
    
    
    ### Task 2.1 ###
    img_a_1_1 = pointToPointRecovery(img_a, corners_a , corners_a_world)
    img_b_1_1 = pointToPointRecovery(img_b, corners_b , corners_b_world)
    
    plt.imshow(img_a_1_1)
    plt.figure()
    plt.imshow(img_b_1_1)
    plt.figure()
    
    
    ### Task 2.2 ###
    img_a_1_2_proj, img_a_1_2_aff = twoStepRecovery(img_a, corners_a)
    img_b_1_2_proj, img_b_1_2_aff = twoStepRecovery(img_b, corners_b)

    plt.imshow(img_a_1_2_proj)
    plt.figure()
    plt.imshow(img_a_1_2_aff)
    plt.figure()
    plt.imshow(img_b_1_2_proj)
    plt.figure()
    plt.imshow(img_b_1_2_aff)
    plt.figure()
    
    
    ### Task 2.3 ###
    
    # one more set of points for the 5th orthogonal line pair
    corners_a2 = np.array([[890, 247], [667, 925], [1470, 979], [1630, 180]])
    corners_b2 = np.array([[56, 78], [61, 808], [770, 1106], [848, 360]])
    
    img_a_1_3 = oneStepRecovery(img_a, corners_a , corners_a2)
    img_b_1_3 = oneStepRecovery(img_b, corners_b , corners_b2)

    plt.imshow(img_a_1_3)
    plt.figure()
    plt.imshow(img_b_1_3)
    plt.figure()    
   

