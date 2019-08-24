# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:14:42 2019

@author: Christos
"""

import cv2
from open3d import *
from cv2 import aruco
import yaml
import numpy as np
from auxiliary_functions import *
import time
from random import randint
from matplotlib import pyplot as plt
%matplotlib auto

start = time.time()

# create dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)

# Creating a theoretical board we'll use to calculate marker positions
board = aruco.GridBoard_create(
    markersX=5,
    markersY=7,
    markerLength=0.047,
    markerSeparation=0.012,
    dictionary=aruco_dict)

arucoParams = aruco.DetectorParameters_create()

# find 4 corners of each marker in WCS
a = board.getGridSize()
b = board.getMarkerLength()
c = board.getMarkerSeparation()

markers = a[0]*a[1]
full = (a[1] - 1)*(b + c) + b
real = []
for i in range(markers):
    x = i % a[0]
    y = i // a[0]
    real.append([[x*(b + c), full - y*(b + c), 0], 
                [x*(b + c) + b, full - y*(b + c), 0],
                [x*(b + c) + b , full - y*(b + c) - b, 0],
                [x*(b + c), full - y*(b + c) - b, 0]])
    

# Load precalculated intrinsic parameters
with open('calibration.yaml') as f:
    intrinsic = yaml.load(f)

Vs = []
calib_times = []
est_times = []

# for all food samples
for food_num in range(24):

    # start timing for calibration
    start = time.time()
    cap = cv2.VideoCapture("samples\\normal\\food" + str(food_num+1) +".mp4") 
#    cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 20)
    cameraMatrix, distCoeffs = calibrate_cap(cap, num = 5, step = 70)

    # if calibration is wrong, load precalculated intrinsic
    if cameraMatrix[0,0] > 1800 or cameraMatrix[1,1] > 1800 :
        cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
        distCoeffs = np.array(intrinsic.get('distCoeffs'))

    # stop timing for calibartion
    end = time.time()
    calib_times.append(end-start)

    # start timing for volume estimation
    start = time.time()
    cap = cv2.VideoCapture("samples\\stereo\\food" + str(food_num+1) +".mp4")    
    step = 10
    
    ret, img1 = cap.read()

    # skip some frames
    for i in range(2*step):
        ret, img1 = cap.read()
        
    img = []
    img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    # create a list of frames
    # if we get just two, it's not guaranteed that they 'll give good result
    while ret:
        ret, img1 = cap.read()
        if img1 is None:
                break
        while is_blur(img1, 100):
            ret, img1 = cap.read()
            if img1 is None:
                break
        img.append(img1)
          
        for i in range(step - 1):
            ret, img1 = cap.read()
            
        if img1 is None:
                break


    # interval between the frames of the list that will form the stereo pair
    half = 8

    # initialization of the transformation matrix used in ICP
    current_transformation = np.identity(4)
    
    u = 0
    ut = 0
    repeats = 0
    br = False
    while u < 1:
        
        repeats += 1
        
        if repeats > 10:
            print("Please, record again!")
            br = True
            break
    
        img1 = img[ut]
        if ut+half >= len(img):
            br = True
            break
        img2 = img[ut + half]

        img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(img1, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(img2, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)

        # find rotation and projection matrices
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1,tvec1)))
    
        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))

        # relative R and T between the two frames
        R = np.dot(a2, np.linalg.inv(a1))
        T = - np.dot(R, tvec1) + tvec2

        # find placemat masks
        mask1 = placemat(img1, cameraMatrix, rvec1, tvec1)
        mask2 = placemat(img2, cameraMatrix, rvec2, tvec2)
        
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 15)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 15)

        # keep only the placemats from the two frames
        img1, img2 = cv2.bitwise_and(img1, img1, mask = mask1), cv2.bitwise_and(img2, img2, mask = mask2)

        # frames' size
        h, w = img1.shape[:2]

        # apply rectification to the two frames
        RL, RR, PL, PR, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix, distCoeffs, 
                                                    cameraMatrix, distCoeffs,  (w, h), R, T,
                                                    alpha=-1)#, flags = cv2.CALIB_ZERO_DISPARITY)

        # If they rotated, continue and take the next pair with higher interval between the frames
        if np.abs(RL[0,0]) < 0.95:
            half += 1
            continue
                      
        mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RL, PL, (w,h), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RR, PR, (w,h), cv2.CV_32FC1)
    
        undistorted_rectifiedL = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR)

        # take the actual rectified images
        imgL, imgR = undistorted_rectifiedL,undistorted_rectifiedR

        # pose estimation
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)

        # during rectification the images may be mirrored, thus the aruco markers can't be detected
        # if that's the case just flip the images
        if rvec1 is None:
            imgL = cv2.flip(imgL, 1 )
            _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        
        if rvec2 is None:
            imgR = cv2.flip(imgR, 1 ) 
            _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
            

        # Find maximum horizontal translation based on the known markers' corners
        fl = False
        dist = []
        for k in range(len(ids1)):
            idd = ids1[k][0]
            if idd not in ids2:
                continue
            idd2 = np.where(ids2 == idd)[0][0]
            sss = np.abs(cor1[k][0,:,0] - cor2[idd2][0,:,0]).max()
            dist.append(sss)
           
        sss = max(dist)

        # if this maximum horizontal translation is higher or lower than some thresholds, continue and get the next pair
        # Also, change interval between the two frames
        if sss > 250:
            half = half - 1
            fl = True

        elif sss < 130:
            half = half + 1
            fl = True

        # Based on this maximum horizontal translation, we specify the num_disp for the SGBM
        n = np.ceil(sss/16) + 1 #max(np.ceil(sss/16) + 2, 15)
        left_matcher.setNumDisparities(int(n)*16)
        right_matcher.setNumDisparities(int(n)*16)
        
        window_size = 7
        min_disp = -1
        num_disp = int(n)*16
        left_matcher = cv2.StereoSGBM_create(
            minDisparity = min_disp,
            numDisparities=num_disp,  
            blockSize=window_size,
            P1=10 * 3 * window_size**2,
            P2=35 * 3 * window_size**2,
            disp12MaxDiff = -1,
            uniquenessRatio=10,
            speckleWindowSize=50,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3
        visual_multiplier = 6
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)
        wls_filter.setSigmaColor(sigma)
        
        
        if  np.abs(RL[2,0]) > 0.15:
            ut += 1
            continue
       
       
    #    fig, ax = plt.subplots(nrows=2, ncols=2)
    #    
    #    plt.subplot(2, 2, 1)
    #    plt.imshow(img1)
    #    
    #    plt.subplot(2, 2, 2)
    #    plt.imshow(img2)
    #    
    #    plt.subplot(2, 2, 3)
    #    plt.imshow(imgL)
    #    
    #    plt.subplot(2, 2, 4)
    #    plt.imshow(imgR)
    

        # find rotation and projection matrices of the rectified images
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1,tvec1)))
        
        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))
        
        R1 = np.dot(a2, np.linalg.inv(a1))
        T1 = - np.dot(R1, tvec1) + tvec2

        # find placemat, food and dish masks
        mask1 = placemat(imgL, cameraMatrix, rvec1, tvec1)
        dish1, d1 = food(imgL, mask1, cor1)
        
        mask2 = placemat(imgR, cameraMatrix, rvec2, tvec2)
        dish2, d2 = food(imgR, mask2, cor2)


        only_plate = cv2.bitwise_and(imgL, imgL, mask = mask1)
        only_plate2 = cv2.bitwise_and(imgR, imgR, mask = mask2)
        
        only_placemat1 = cv2.bitwise_and(mask1, mask1, mask = cv2.bitwise_not(d1))
        only_placemat2 = cv2.bitwise_and(mask2, mask2, mask = cv2.bitwise_not(d2))

        # calculate disparity map
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
            
        disparity = filteredImg
      
#        plt.figure()
#        plt.imshow(disparity,'gray')
#        plt.show()

        # normalize disparity map in order to represent the horizontal translation in pixels
        dmod = []
        h1 = 0
        while h1 < len(ids1):
            idd = ids1[h1][0]
            if idd not in ids2:
                h1+=1
                continue
            idd2 = np.where(ids2 == idd)[0][0]
            sss = np.abs(cor1[h1][0,:,0] - cor2[idd2][0,:,0])

            
            c1 = np.array(np.round(cor1[h1].reshape(4,2)), dtype = int)
            
            if np.where(c1[:,1] >= h)[0].shape[0] > 0 or np.where(c1[:,0] >= w)[0].shape[0] > 0 :
                h1 += 1
                continue
            
            d1 = disparity[c1[0,1],c1[0,0]]
            d2 = disparity[c1[1,1],c1[1,0]]
            d3 = disparity[c1[2,1],c1[2,0]]
            d4 = disparity[c1[3,1],c1[3,0]]
            
            d1 = sss[0]/d1
            d2 = sss[1]/d2
            d3 = sss[2]/d3
            d4 = sss[3]/d4
            
            if d1 < np.inf:
                dmod.append(d1)
            if d2 < np.inf:
                dmod.append(d2)
            if d3 < np.inf:
                dmod.append(d3)
            if d4 < np.inf:
                dmod.append(d4)

            h1 += 1
         
        dxxx = np.round([1000*t for t in dmod])/1000
        (values,counts) = np.unique(dxxx,return_counts=True)
        ind=np.argmax(counts)
        s = values[ind]
        
        # get disparity for food pixels
        d = disparity[dish1>0]

        # some heuristics regarding the quality of the disparity map
        if np.unique(d).shape[0] > 120 :
            half = half - 2
            continue
        elif np.unique(d).shape[0] < 10:#15
            ut += 1
            continue

        d = d*s
        # food pixels
        rows, cols = np.where(dish1 > 0)

        # get reconstructed points in camera 1 coordinate system
        with np.errstate(divide='ignore',invalid='ignore'):
            pw = 1/(d*Q[3,2] + Q[3,3])
        
        X = (cols)*pw
        Y = (rows)*pw
        Z = Q[2,3]*pw
     
        # get colors of these points just for visualization
        only_plate = cv2.bitwise_and(imgL, imgL, mask = dish1)
        colors = cv2.cvtColor(only_plate, cv2.COLOR_BGR2RGB)
        c = colors[np.where(dish1>0)]

        # get rid of some points that are at infitiy because they had 0 in disparity map
        ind = np.where((Z < np.inf) & (Z > -np.inf) & (X > -np.inf) & (X < np.inf) & (Y > -np.inf) & (Y < np.inf))
        X = X[ind]
        Y = Y[ind]
        Z = Z[ind]
        c = c[ind]

        # choose 5000 random points
        f = [randint(0, len(X)-1) for q in range(0, 5000)]
        X = X[f]
        Y = Y[f]
        Z = Z[f]
        c = c[f]

        #  go from camera1 coordinate system to WCS
        p = np.array([X, Y, Z])
        pw = np.zeros((3,len(X)))
        a = np.linalg.inv(a1)
        b = np.dot(np.linalg.inv(a1), tvec1)
        for k in range(len(X)):
            pw[:,k] = np.diag(np.dot(a, p[:,k]) - b)
        X = pw[0,:]
        Y = pw[1,:]
        Z = pw[2,:]

        C = c/255.0

        # find placemat coordintes
        d = disparity[only_placemat1>0]    
        rows, cols = np.where(only_placemat1 > 0)
        d = d*s
        with np.errstate(divide='ignore',invalid='ignore'):
            pw = 1/(d*Q[3,2] + Q[3,3])
        
        X_placemat = (cols)*pw
        Y_placemat = (rows)*pw
        Z_placemat = Q[2,3]*pw

        ind = np.where((Z_placemat < np.inf) & (Z_placemat > -np.inf) & 
                       (X_placemat > -np.inf) & (X_placemat < np.inf) & 
                       (Y_placemat > -np.inf) & (Y_placemat < np.inf))
        X_placemat = X_placemat[ind]
        Y_placemat = Y_placemat[ind]
        Z_placemat = Z_placemat[ind]
        
    
        f = [randint(0, len(X_placemat)-1) for q in range(0, 5000)]
        X_placemat = X_placemat[f]
        Y_placemat = Y_placemat[f]
        Z_placemat = Z_placemat[f]
      
    
        #  placemate coords from camera 1 to  WCS
        p = np.array([X_placemat, Y_placemat, Z_placemat])
        pw = np.zeros((3,len(X_placemat)))
        a = np.linalg.inv(a1)
        b = np.dot(np.linalg.inv(a1), tvec1)
        for k in range(len(X_placemat)):
            pw[:,k] = np.diag(np.dot(a, p[:,k]) - b)
        X_placemat = pw[0,:]
        Y_placemat = pw[1,:]
        Z_placemat = pw[2,:]


        # find median of placemat Z coordinate as the placemat level
        placemat_level = np.median(Z_placemat)
        
        # get rid of points below placemat level
        ind = np.where(Z > placemat_level)
        X = X[ind]
        Y = Y[ind]
        Z = Z[ind]
        C = C[ind]
        
        # align to (0,0,0)
        X = X - np.min(X)
        Y = Y - np.min(Y)
        Z = Z - placemat_level
        

        u += 1
    
    # dish height
    bottom =  0.009

    # keep points that are higher than dish height
    ind = np.where((Z < 0.20) & (Z > bottom) & (X > 0) & (X < 0.5) & (Y > 0) & (Y < 0.5))
    X = X[ind]
    Y = Y[ind]
    Z = Z[ind]
    C = C[ind]

    # create point cloud
    xyz = np.vstack([X,Y,Z]).transpose()
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(C)
    
    # Downsample to voxel
    voxel = 0.005
    downpcd = voxel_down_sample(pcd, voxel_size = voxel)
    
    
    # Outlier removal
    cl1,ind = radius_outlier_removal(downpcd,
            nb_points=15, radius=3*voxel)
        
    inlier_cloud = select_down_sample(downpcd, ind)
    inliers = np.asarray(inlier_cloud.points)
            
    X = inliers[:,0]
    Y = inliers[:,1]
    Z = inliers[:,2]
    C = np.asarray(inlier_cloud.colors)

    # Plot point cloud
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    scat = ax.scatter(X, Y, Z, c = C, s = 3)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()
    
    
    # Calculate volume
    vol = int(1000*volume_estimation(X, Y, Z, bottom, voxel, avg_dist = 4))
#    print("Volume is: ", vol)
    
    if br:
        vol = 0
        
    Vs.append(vol)

    # stop timing for volume estimation
    end = time.time()
#    print("Time elapsed: ", end - start)

    est_times.append(end-start)