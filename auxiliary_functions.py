# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:34:09 2019

@author: Xristos
"""

import cv2
from cv2 import aruco
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull



def calibrate_cap1(cap, step):
    """ Calibrates the camera for intrinsic parameters.
    
    Parameters:
        cap : The video object.
        num: Number of frames that will be used for calibration.
        step: Distance between frames.
        
    Returns:
        cameraMatrix: Camera matrix.
        distCoeffs: Distortion coefficients. 
    """
    
    # create dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

    # Creating a theoretical board we'll use to calculate marker positions
    board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=0.047,
        markerSeparation=0.012,
        dictionary=aruco_dict)
    
    # Initialize parameters and criteria for markers' detection
    parameters = aruco.DetectorParameters_create()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3) 
        
    counter = []
    corners_list = []
    id_list = []
    first = True
    ret, img = cap.read()
    # For each image detect markers.
    while ret == True:
        
        for t in range(step):
            ret , img2 = cap.read()
        
        ret, img = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        for k in range(len(corners)):
            cv2.cornerSubPix(gray, corners[k], (7, 7), (-1,-1), criteria)
     
        # Put each marker's id and corners in seperate lists,
        # along with the number of markers
        if len(ids) > 1:
            if first == True:
                corners_list = corners
                print(type(corners))
                id_list = ids
                first = False
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
                 
            counter.append(len(ids))
           
            
    # Calibrate
    counter = np.array(counter)    
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, gray.shape, None, None )
    return cameraMatrix, distCoeffs
    


def calibrate_cap(cap, num, step):
    """ Calibrates the camera for intrinsic parameters.
    
    Parameters:
        cap : The video object.
        num: Number of frames that will be used for calibration.
        step: Distance between frames.
        
    Returns:
        cameraMatrix: Camera matrix.
        distCoeffs: Distortion coefficients. 
    """
    
    # create dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

    # Creating a theoretical board we'll use to calculate marker positions
    board = aruco.GridBoard_create(
        markersX=5,
        markersY=7,
        markerLength=0.047,
        markerSeparation=0.012,
        dictionary=aruco_dict)
    
    # Initialize parameters and criteria for markers' detection
    parameters = aruco.DetectorParameters_create()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3) 
        
    counter = []
    corners_list = []
    id_list = []
    first = True

    # For each image detect markers.
    for i in range(num):
        
        for t in range(step):
            ret , img2 = cap.read()
        
        ret, img = cap.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        for k in range(len(corners)):
            cv2.cornerSubPix(gray, corners[k], (7, 7), (-1,-1), criteria)
     
        # Put each marker's id and corners in seperate lists,
        # along with the number of markers
        if len(ids) > 1:
            if first == True:
                corners_list = corners
                print(type(corners))
                id_list = ids
                first = False
            else:
                corners_list = np.vstack((corners_list, corners))
                id_list = np.vstack((id_list,ids))
                 
            counter.append(len(ids))
           
            
    # Calibrate
    counter = np.array(counter)    
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, gray.shape, None, None )
    return cameraMatrix, distCoeffs
    


def pose_estimation(img, board, aruco_dict, arucoParams, mtx, dist):
    """ Estimates the R and T matrices from aruco board.
    
    Parameters:
        img : The original image.
        board: Aruco board object.
        aruco_dict: Dictionary for aruco markers.
        arucoParams: Parameters for aruco.
        mtx: The camera matrix.
        dist: Distortion coefficients.
        
    Returns:
        ret: Value set as 1 if the pose is estimated, 0 otherwise. 
        rvec: Rotation vector of camera.
        tvec: Translation vector of camera.
        corners: Pixels where the aruco markers' corners where found.
        ids: List of markers' ids.
        
    """
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3) 
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
    for k in range(len(corners)):
            cv2.cornerSubPix(img_gray, corners[k], (5, 5), (-1,-1), criteria)
    if corners == None:
        print ("No markers detected. Can't estimate pose.")
        return 0, None, None, None, None
    else:
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, mtx, dist)
        return 1, rvec, tvec, corners, ids
    
    
    

    
def placemat(img, mtx, rvec, tvec):
    
    """ Returns a mask with placemat pixels.
    
    Parameters:
        img : The original image where we will track the placemat.
        mtx: The camera matrix.
        rvec: Rotation vector as returned from 'pose_estimation'.
        tvec: Translation vector as returned from 'pose_estimation'.
        
    Returns:
        A mask with the size of the original image, with every placemat pixel set to 1
        and every other pixel set to 0.
        
    """
    
    # Boards corners in 3D WCS
    borders = np.array([[0, 0, 0, 1], [0.283, 0, 0, 1], [0.283, 0.401, 0, 1], [0, 0.401, 0, 1]])
    borders = borders.transpose()
    
    # R matrix from r vector
    a, _ = cv2.Rodrigues(rvec)
    # Projection matrix
    L = np.dot(mtx, np.hstack((a,tvec)))

    # Board's corners in CCS
    borders_in_image = np.dot(L,borders)
   
    # Board's corners projection in image
    s = borders_in_image[2, ...]
    x = np.round(borders_in_image[0, ...]/s)
    x = x.astype(int)

    y = np.round(borders_in_image[1, ...]/s)
    y = y.astype(int)

    # stack all 4 corners in a matrix
    a = np.vstack((x,y))
    a = a.transpose()

    b = img.shape[:2]


    # Create the smallest possible image that is at least the size of the original
    # image, but also has the entire placemat in it (regardless if it is visible in the image)
    x1 = min(min(x), 0)
    x2 = max(max(x), b[1])

    y1 = min(min(y), 0)
    y2 = max(max(y), b[0])

    size = tuple([abs(y2-y1) + 1, abs(x2-x1) + 1])

    k = np.zeros(size, np.uint8)

    a[..., 0] = a[..., 0] - x1
    a[..., 1] = a[..., 1] - y1

    # Fill the area of placemat with white
    k = cv2.fillPoly(k, np.array([a], dtype=np.int32), 255)
    
    # Crop the mask in order to match the original
    mask = k[abs(y1):b[0]+abs(y1), abs(x1):b[1] + abs(x1)]
    
    return mask







def food(image, placemat, cor):
    """ Returns two mask with food and dish pixels 
    
    Parameters:
        image : The original image where we will track the food.
        placemat: The placemat mask as returned from function 'placemat'.
        cor: corners of aruco markers as returned from 'pose_estimation'.
        
    Returns:
        Two masks with the size of the original image, with every food/dish pixel as 1
        and every other pixel as 0.
        
    """
    # Image's size
    h, w = image.shape[0:2]
    # Create a mask with all aruco markers
    markers_mask = np.zeros([h,w], np.uint8)
    for i in range(len(cor)):
        markers_mask = cv2.fillPoly(markers_mask, np.array(cor[i], dtype=np.int32), 255)


    # Keep only the placemat through the mask
    only_placemat = cv2.bitwise_and(image, image, mask = placemat)
    
    # Convert to HSV colorspace
    hsv = cv2.cvtColor(only_placemat,cv2.COLOR_RGB2HSV)


    # Calclate the histogram of the image in the hsv colorspace ONLY for the markers
    roihist = cv2.calcHist([hsv],[0, 1], markers_mask, [180, 256], [0, 180, 0, 256] )

    # Values for H in the placemat
    h1 = 92 #apo 94 stis 19/6 gia to food23 pou eixe h = 102
    h2 = 163
    
    # Find the most common S value for each H
    s1 = np.where(roihist[85:101]==roihist[85:101].max())[1][0]
    s2 = np.where(roihist[150:180]==roihist[150:180].max())[1][0]
    
    # Set some limits for H ans S for both colors and find pixels in these limits
    upper1 =  np.array([h1 + 8, s1 + 50, 255])
    lower1 =  np.array([h1 - 8, s1 - 30, 0])
    c1 = cv2.inRange(hsv, lower1, upper1)
     
    upper2 =  np.array([h2 + 15, s2 + 70, 255])
    lower2 =  np.array([h2 - 15, s2 - 50, 0])
    c2 = cv2.inRange(hsv, lower2, upper2)
    
    # Combine the results in color_mask
    color_mask = cv2.bitwise_or(c1, c2)

#
#    # Some weird greenish colors in HSV
#    if roihist[0:20, 1:256].max() > 800:
#        upper3 =  np.array([20, 180, 255])
#        lower3 =  np.array([0, 0, 0])
#        c3 = cv2.inRange(hsv, lower3, upper3)
#        color_mask = cv2.bitwise_or(color_mask, c3)


    # Fill the markers for potential missing pixels in color_mask
    for i in range(len(cor)):
        color_mask = cv2.fillPoly(color_mask, np.array(cor[i], dtype=np.int32), 255)

    # Use morphological close for missing pixels
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations = 3)
  
    # Reverse the mask so that dish has value 1 and the rest placemat 0
    color_mask = cv2.bitwise_not(color_mask)

    placemat = cv2.morphologyEx(placemat, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 30)
    # The area beyond placemat is set to 0
    dish_mask = cv2.bitwise_and(color_mask, placemat)


## apo 20 se 15   18/6 gia araka
#    dish_mask = cv2.morphologyEx(dish_mask, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 5) 
    dish_mask = cv2.morphologyEx(dish_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations = 10) # apo 15 se 10 stis 19/6
    
    
    
    im_floodfill = dish_mask.copy()
 
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = dish_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = dish_mask | im_floodfill_inv
    
    # Some morphological operations
    im_out = cv2.morphologyEx(im_out, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 20)
    dish = cv2.morphologyEx(im_out, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 20)
   
    
   # Find contours
    _, contours, _ = cv2.findContours(dish, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
    # Draw contours + hull results
    drawing = np.zeros((dish.shape[0], dish.shape[1]), dtype=np.uint8)
    for i in range(len(contours)):
        color = 255
        cv2.drawContours(drawing, contours, i, color)
        cv2.drawContours(drawing, hull_list, i, color)

    d = cv2.morphologyEx(drawing, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 3)
    
    im_floodfill = d.copy()
    
    h, w = d.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = d | im_floodfill_inv
    
    dish = im_out
    
    
    # Mask the original image in order to have just the dish 
    plate = cv2.bitwise_and(image, image, mask = im_out)

#    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2HSV)

    gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    
    # Find edges
    ret, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    
    
    ########### 13/6 apo 1.25*ret to piga 0.5*ret gia kaluteres (pio euaisthito)akmes sta kolokythakia
    lim = 1.3
   
    edg = cv2.Canny(gray, 30, 80)
    #############

    # Find the contours that are formed from the edges
    th1 = gray*edg
    _, contours,hierarchy = cv2.findContours(th1,2,1)


    img = np.zeros(gray.shape, image.dtype)
    cv2.drawContours(img, contours, -1, 255, 3)
           
    e = cv2.morphologyEx(im_out, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 15)
    cl = e*img

    cl = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations = 10)
    
    ## 12/6 arakas
    cl = cv2.morphologyEx(cl, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 4)
    ###


    # Copy the thresholded image.
    im_floodfill = cl.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = cl.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    food = cl | im_floodfill_inv
     

    cl = cv2.morphologyEx(food, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations = 20)
    food = cv2.morphologyEx(cl, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 2 )
    
    
    ##### fix this
    dif = 100*(np.where(dish>1)[0].shape[0] - np.where(food>1)[0].shape[0])/np.where(dish>1)[0].shape[0]
    if dif < 15:
        lim = 1.8
        edg = cv2.Canny(gray, 0.1*ret, lim*ret)
        #############
    
        # Find the contours that are formed from the edges
        th1 = gray*edg
        _, contours,hierarchy = cv2.findContours(th1,2,1)
    
    
        img = np.zeros(gray.shape, image.dtype)
        cv2.drawContours(img, contours, -1, 255, 3)
               
        e = cv2.morphologyEx(im_out, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 15)
        cl = e*img
    
        cl = cv2.morphologyEx(cl, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations = 10)
        
        ## 12/6 arakas
        cl = cv2.morphologyEx(cl, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 4)
        ###
    
    
        # Copy the thresholded image.
        im_floodfill = cl.copy()
         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = cl.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
         
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
         
        # Combine the two images to get the foreground.
        food = cl | im_floodfill_inv
         
    
        cl = cv2.morphologyEx(food, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations = 20)
        food = cv2.morphologyEx(cl, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 2 )
    ####
        
    return food , dish  


def volume_estimation(X, Y, Z, bottom, voxel, avg_dist):
    """ Estimates the volume of a point cloud through Delaunay triangulation. 
    
    Parameters:
        X, Y, Z: Coordinate vectors.
        bottom: The height of the dish bottom with respect to the placemat.
        voxel: voxel size
        avg_dist: maximum average distance to consider 3 points neighbors in voxel size
    Returns:
        Estimated volume.
        
    """
    
    # Project the real points to the bottom level
    points = np.zeros((len(X),3))
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = bottom
    

    real_points = np.copy(points)
    real_points[:,2] = Z
    
    # Delaunay for the projected points
    tri = Delaunay(points[:,0:2], furthest_site = False)
    
    # For each triangle combine the 3  projected and the 3 real points
    # and form a ConvexHull. Compute the volume and add it to the total volume.
    V = 0
    for i in range(len(tri.simplices)):
        p = points[tri.simplices[i]]
        d1 = np.linalg.norm(p[0]- p[1])
        d2 = np.linalg.norm(p[2]- p[1])
        d3 = np.linalg.norm(p[0]- p[2])
        avg = (d1 + d2 + d3)/3
        # If the average pairwise distance of the projected points is high, 
        # dont compute the volume.
        if avg > avg_dist*voxel:
            continue
        point_set = np.vstack([points[tri.simplices[i]], real_points[tri.simplices[i]]])
        Q = ConvexHull(point_set)
        V = V + Q.volume
    V = V*1000
    
    return V


def is_blur(img, blur_threshold):
    """ Determines if an image is blur. 
    
    Parameters:
        img: The image.
        blur_threshold: Threshold value in order to determine if an image is blur
                        or not. A typical value is 100.
    Returns:
        Boolean True if image is blur, False otherwise.
        
    """    
    flag = False
    if cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold:
        flag = True
    return flag