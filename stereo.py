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
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull


start = time.time()

# Initialize Semi Global Block Matching  
window_size = 5
min_disp = -1
num_disp = 20*16
left_matcher = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities=num_disp,  
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff = -1,
    uniquenessRatio=10,
    speckleWindowSize=40,  #apo 50
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

# get 4 corners of each marker in WCS
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
    

# load intrinsic parameters
with open('calibration.yaml') as f:
    intrinsic = yaml.load(f)
cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
distCoeffs = np.array(intrinsic.get('distCoeffs'))

Vs =[]
for uuu in range(5):
    
    
    vid = "samples\\normal\\food13.mp4"
    cap = cv2.VideoCapture(vid)
    cameraMatrix, distCoeffs = calibrate_cap(cap, num = 5, step = 70)
#    cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 100)
    if cameraMatrix[0,0] > 1800 or cameraMatrix[1,1] > 1800 :
        with open('calibration.yaml') as f:
            intrinsic = yaml.load(f)  
        cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
        distCoeffs = np.array(intrinsic.get('distCoeffs'))
        
    
    start = time.time()
    cap = cv2.VideoCapture("samples\\stereo\\food13.mp4")
    step = 10
    
    ret, img1 = cap.read()
    
    for i in range(uuu*step):
        ret, img1 = cap.read()
        
    img = []
    img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    while ret:
        ret, img1 = cap.read()
        while is_blur(img1, 100):
            ret, img1 = cap.read()
            if img1 is None:
                break
        img.append(img1)
          
        for i in range(step - 1):
            ret, img1 = cap.read()
            
        if img1 is None:
                break
    
    half = 8
    
    current_transformation = np.identity(4)
    
    u = 0
    ut = 0
    
    #cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, img_gray.shape[::-1], 0, img_gray.shape[::-1])
    
    repeats = 0
    
    while u < 1:
        
        repeats += 1
        
        if repeats > 10:
            print("Please, record again!")
            break
    
        img1 = img[ut]
        img2 = img[ut + half]
        
    #    img1 = cv2.undistort(img1, cameraMatrix, distCoeffs, None, cameraMatrix)
    #    img2 = cv2.undistort(img2, cameraMatrix, distCoeffs, None, cameraMatrix)
        
        img_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(img1, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(img2, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        
        
#        p1 = []
#        p2 = []
#        obj =[]
#        
#        for j in range(len(ids1)):
#            if ids1[j] in ids2:
#                th = np.where(ids2 == ids1[j])[0]
#                for k in range(4):
#                    p1.append(cor1[j][0][k])
#                    p2.append(cor2[th[0]][0][k])
#                    obj.append(real[ids1[j][0]][0])
#                
#                
#        obj = np.array(obj, np.float32).reshape(1,-1,3)
#        p1 = np.array(p1, np.float32).reshape(1,-1,2)
#        p2 = np.array(p2, np.float32).reshape(1,-1,2)
#          
#        stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
#        stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
#        
#        flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_FOCAL_LENGTH |
#                 cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |
#                 cv2.CALIB_FIX_K6)
#        
#        
#        stereocalibration_retval, cameraMatrix1, distCoeffs1, cameraMatrix2,distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj,p1,p2,
#             cameraMatrix,distCoeffs,cameraMatrix,distCoeffs,img_gray.shape[::-1],
#             criteria = stereocalibration_criteria, flags = flags)
#        
    
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1,tvec1)))
    
        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))
#    
        R = np.dot(a2, np.linalg.inv(a1))
        T = - np.dot(R, tvec1) + tvec2
        
        mask1 = placemat(img1, cameraMatrix, rvec1, tvec1)
        mask2 = placemat(img2, cameraMatrix, rvec2, tvec2)
        
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 15)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_DILATE, np.ones((3,3),np.uint8), iterations = 15)
        
        img1, img2 = cv2.bitwise_and(img1, img1, mask = mask1), cv2.bitwise_and(img2, img2, mask = mask2)
    
        h, w = img1.shape[:2]
           
        RL, RR, PL, PR, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix, distCoeffs, 
                                                    cameraMatrix, distCoeffs,  (w, h), R, T,
                                                    alpha=-1)#, flags = cv2.CALIB_ZERO_DISPARITY)
        
        if np.abs(RL[0,0]) < 0.95:
            half += 1
            continue
##        
#        if np.abs(RL[0,1]) > 0.1:
#            ut += 1
#            half+=1
#            continue
#        
    #    if np.abs(RL[1,0]) > 0.1 or np.abs(RL[2,0]) > 0.13:
    #        ut += 1
    #        half += 1
    #        continue
                      
        mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RL, PL, (w,h), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RR, PR, (w,h), cv2.CV_32FC1)
    
        undistorted_rectifiedL = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR)
        
        imgL, imgR = undistorted_rectifiedL,undistorted_rectifiedR
           
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
    
        if rvec1 is None:
            imgL = cv2.flip(imgL, 1 )
            _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        
        if rvec2 is None:
            imgR = cv2.flip(imgR, 1 ) 
            _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
            
            
        if ids1 is None or ids2 is None:
            ut+=1
            continue
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
        if sss > 250:
            half = half - 1
            fl = True
#            break
        elif sss < 130:
            half = half + 1
            fl = True
#            break
#      
#        if fl:
#            continue
#                
            
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
            P1=8 * 3 * window_size**2,
            P2=32 * 3 * window_size**2,
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
    #        half += 1
            continue
       
       
#        fig, ax = plt.subplots(nrows=2, ncols=2)
#        
#        plt.subplot(2, 2, 1)
#        plt.imshow(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))
#        
#        plt.subplot(2, 2, 2)
#        plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
#        
#        plt.subplot(2, 2, 3)
#        plt.imshow(cv2.cvtColor(imgL,cv2.COLOR_BGR2RGB))
#        
#        plt.subplot(2, 2, 4)
#        plt.imshow(cv2.cvtColor(imgR,cv2.COLOR_BGR2RGB))
    
        
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1,tvec1)))
        
        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))
        
        R1 = np.dot(a2, np.linalg.inv(a1))
        T1 = - np.dot(R1, tvec1) + tvec2
        
        mask1 = placemat(imgL, cameraMatrix, rvec1, tvec1)
        dish1, d1 = food(imgL, mask1, cor1)
        
        mask2 = placemat(imgR, cameraMatrix, rvec2, tvec2)
        dish2, d2 = food(imgR, mask2, cor2)
        
        only_plate = cv2.bitwise_and(imgL, imgL, mask = mask1)
        only_plate2 = cv2.bitwise_and(imgR, imgR, mask = mask2)
        
    #    d1 = dish(imgL, mask1, cor1)
    #    d2 = dish(imgR, mask2,cor2)
        
        only_placemat1 = cv2.bitwise_and(mask1, mask1, mask = cv2.bitwise_not(d1))
        only_placemat2 = cv2.bitwise_and(mask2, mask2, mask = cv2.bitwise_not(d2))
        

        
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  
        
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
            
        disparity = filteredImg
      
        plt.figure()
        plt.imshow(disparity,'gray')
        plt.show()
             
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
            
            dmod.append(d1)
            dmod.append(d2)
            dmod.append(d3)
            dmod.append(d4)
            
           
            
            h1 += 1
         
        dxxx = np.round([1000*t for t in dmod])/1000
        (values,counts) = np.unique(dxxx,return_counts=True)
        ind=np.argmax(counts)
        s = values[ind]
        
        
#        disparity = disparity*s
        
#        dish1 = cv2.morphologyEx(dish1, cv2.MORPH_ERODE, np.ones((3,3),np.uint8), iterations = 5)
        # food coords  
    
        d = disparity[dish1>0]    
        if np.unique(d).shape[0] > 120 :
            half = half - 2
            continue
        elif np.unique(d).shape[0] < 10:
            ut += 1
            continue
        if np.median(d) < 200:
            ut += 1
            continue
        rows, cols = np.where(dish1 > 0)
        
        d = d*s
    
        with np.errstate(divide='ignore',invalid='ignore'):
            pw = 1/(d*Q[3,2] + Q[3,3])
        
        X = (cols)*pw
        Y = (rows)*pw
        Z = Q[2,3]*pw
     
        
        only_plate = cv2.bitwise_and(imgL, imgL, mask = dish1)
        colors = cv2.cvtColor(only_plate, cv2.COLOR_BGR2RGB)
        c = colors[np.where(dish1>0)]
        
#        # scaling factors         
#        a = np.linalg.inv(a1)
#        b = np.dot(np.linalg.inv(a1), tvec1)
#        
#        dxx = []
#        dyy = []
#        theta = []
#        h1 = 0
#        while h1 < len(ids1):
#            c1 = np.array(np.round(cor1[h1].reshape(4,2)), dtype = int)
#            
#            if np.where(c1[:,1] >= h)[0].shape[0] > 0 or np.where(c1[:,0] >= w)[0].shape[0] > 0 :
#                h1 += 1
#                continue
#            
#            d1 = disparity[c1[0,1],c1[0,0]]
#            d2 = disparity[c1[1,1],c1[1,0]]
#            d3 = disparity[c1[2,1],c1[2,0]]
#            d4 = disparity[c1[3,1],c1[3,0]]
#            
#            with np.errstate(divide='ignore',invalid='ignore'):
#                pw1 = 1/(d1*Q[3,2] + Q[3,3])
#                pw2 = 1/(d2*Q[3,2] + Q[3,3])
#                pw3 = 1/(d3*Q[3,2] + Q[3,3])
#                pw4 = 1/(d4*Q[3,2] + Q[3,3])
#            
#            x1 = np.array([(c1[0,0])*pw1, (c1[0,1])*pw1, Q[2,3]*pw1])
#            x2 = np.array([(c1[1,0])*pw2, (c1[1,1])*pw2, Q[2,3]*pw2])
#            x3 = np.array([(c1[2,0])*pw3, (c1[2,1])*pw3, Q[2,3]*pw3])
#            x4 = np.array([(c1[3,0])*pw4, (c1[3,1])*pw4, Q[2,3]*pw4])
#      
#            x11 = np.diag(np.dot(a, x1) - b)
#            x22 = np.diag(np.dot(a, x2) - b)
#            x33 = np.diag(np.dot(a, x3) - b)
#            x44 = np.diag(np.dot(a, x4) - b)
#            
#            theta.append(90 - np.arctan((x22[1]-x11[1])/(x22[0]-x11[0]))*180/np.pi)
#        
#            dx =  np.linalg.norm(x11[0:3]-x22[0:3])
#            dy =  np.linalg.norm(x33[0:3]-x22[0:3])
#            
#            dx2 =  np.linalg.norm(x33[0:3]-x44[0:3])
#            dy2 =  np.linalg.norm(x11[0:3]-x44[0:3])
#            
#            
#            if np.isnan(dx) or np.isnan(dy) or np.isnan(dx2) or np.isnan(dy2):
#                h1 = h1 + 1
#                continue
#            
#            dxx.append(dx)
#            dyy.append(dy)
#            dxx.append(dx2)
#            dyy.append(dy2)
##            dyy.append(dy)
#            
#            h1 += 1
#    
#        
#        dxxx = np.round([1000*t for t in dxx])/1000
#        (values,counts) = np.unique(dxxx,return_counts=True)
#        ind=np.argmax(counts)
#        s = values[ind]
#        
##        s = np.mean(dxxx)
##        s = np.median(dxxx)
#      
#        s = board.getMarkerLength()/s
#        
        
        # downsample    
        ind = np.where((Z < np.inf) & (Z > -np.inf) & (X > -np.inf) & (X < np.inf) & (Y > -np.inf) & (Y < np.inf))
        X = X[ind]
        Y = Y[ind]
        Z = Z[ind]
        c = c[ind]
    
        f = [randint(0, len(X)-1) for q in range(0, 5000)]
        X = X[f]
        Y = Y[f]
        Z = Z[f]
        c = c[f]
      
    
        #  ~ WCS
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
        
    
        # placemat coords
        d = disparity[only_placemat1>0]    
        rows, cols = np.where(only_placemat1 > 0)
        
        d = d*s
    
        with np.errstate(divide='ignore',invalid='ignore'):
            pw = 1/(d*Q[3,2] + Q[3,3])
        
        X_placemat = (cols)*pw
        Y_placemat = (rows)*pw
        Z_placemat = Q[2,3]*pw
        
        # downsample    
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
      
    
        #  ~ WCS
        p = np.array([X_placemat, Y_placemat, Z_placemat])
        pw = np.zeros((3,len(X_placemat)))
        a = np.linalg.inv(a1)
        b = np.dot(np.linalg.inv(a1), tvec1)
        for k in range(len(X_placemat)):
            pw[:,k] = np.diag(np.dot(a, p[:,k]) - b)
        X_placemat = pw[0,:]
        Y_placemat = pw[1,:]
        Z_placemat = pw[2,:]
     
        placemat_level = np.median(Z_placemat)
        
        
        
        ind = np.where(Z > placemat_level)
        X = X[ind]
        Y = Y[ind]
        Z = Z[ind]
        C = C[ind]
        
#        ind = np.where(Z_placemat > placemat_level - 0.001)
#        X_placemat = X_placemat[ind]
#        Y_placemat = Y_placemat[ind]
#        Z_placemat = Z_placemat[ind]
#
#        Z_placemat = s*Z_placemat
#        X_placemat = s*X_placemat
#        Y_placemat = s*Y_placemat
#        
#        xmin = X_placemat.min()
#        ymin = Y_placemat.min()

        

        
        
        # align to (0,0,-) and actual scale 
        st = np.array([0, 0, 0, 1])
        st = st.transpose()
    
        # Board's corners in CCS
        st_in_image = np.dot(np.hstack((a1,tvec1)),st)
        
#        X = X + st_in_image[0]
#        Y = Y + st_in_image[1]
        Z = Z - placemat_level
        
        th = np.median(theta)
#        s = s*np.abs(np.cos(th*np.pi/180))
#        rot_mat = np.array([[np.cos(th*np.pi/180), np.sin(th*np.pi/180)],
#                            [-np.sin(th*np.pi/180), np.cos(th*np.pi/180)]])
#        
#        new_xy = np.matmul(rot_mat, np.vstack([X,Y]))
#        X = new_xy[0]
#        Y = new_xy[1]
#        
        X = X - X.min()
        Y = Y -  Y.min()
#        
#        Z = s*Z
#        X = s*X
#        Y = s*Y
#        
        u += 1
        
        Z_placemat = Z_placemat - placemat_level
    
        
        ind = np.where((Z_placemat >  - 0.1) & (Z_placemat < 0.1))
        X_placemat = X_placemat[ind]
        Y_placemat = Y_placemat[ind]
        Z_placemat = Z_placemat[ind]
     
    
#      
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        ax.set_aspect('equal')
#        scat = ax.scatter(X, Y, Z, c = C, s = 3)
#        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#        mid_x = (X.max()+X.min()) * 0.5
#        mid_y = (Y.max()+Y.min()) * 0.5
#        mid_z = (Z.max()+Z.min()) * 0.5
#        ax.set_xlim(mid_x - max_range, mid_x + max_range)
#        ax.set_ylim(mid_y - max_range, mid_y + max_range)
#        ax.set_zlim(mid_z - max_range, mid_z + max_range)
#        plt.show()
#        
#        
#        new_xy = np.matmul(rot_mat, np.vstack([X_placemat,Y_placemat]))
#        X_placemat = new_xy[0]
#        Y_placemat = new_xy[1]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect('equal')
        scat = ax.scatter(X_placemat, Y_placemat, Z_placemat, s = 3)
        max_range = np.array([X_placemat.max()-X_placemat.min(), Y_placemat.max()-Y_placemat.min(), Z_placemat.max()-Z_placemat.min()]).max() / 2.0
        mid_x = (X_placemat.max()+X_placemat.min()) * 0.5
        mid_y = (Y_placemat.max()+Y_placemat.min()) * 0.5
        mid_z = (Z_placemat.max()+Z_placemat.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show()
#        
#    
#        
        
    
    
    if repeats >10:
        continue
    
    bottom =  0.009 # max(0, Zs.min())
    
    ind = np.where((Z < 0.20) & (Z > bottom) & (X > 0) & (X < 0.5) & (Y > 0) & (Y < 0.5))
    
    X = X[ind]
    Y = Y[ind]
    Z = Z[ind]
    C = C[ind]
    
#    
#    final = pcd
#    
    xyz = np.vstack([X,Y,Z]).transpose()
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(C)
    
#    
#    tr = registration_icp(pcd, final, 0.01, current_transformation,
#            TransformationEstimationPointToPoint())#,
##            ICPConvergenceCriteria(max_iteration = 2000))
#    pcd.transform(tr.transformation)
#    t = np.asarray(final.points)
#    col = np.asarray(final.colors)
#    s = np.asarray(pcd.points)
#    xyz = np.vstack([t,s])
#    final = PointCloud()
#    final.points = Vector3dVector(xyz)
#    final.colors = Vector3dVector(np.vstack([col, C]))
#
#
#    pcd = final
    # Downsampling
    voxel = 0.005
    downpcd = voxel_down_sample(pcd, voxel_size = voxel)
    
#    
#    # Outlier removal
    cl1,ind = radius_outlier_removal(downpcd,
            nb_points=15, radius=3*voxel)
    
#    if len(ind)/np.asarray(downpcd.points).shape[0] < 0.8:
#        cl1,ind = radius_outlier_removal(downpcd,
#        nb_points=15, radius=3*voxel)
#    
#    
        
    
#    neighbs = 15
#    while True:
#        cl1,ind = radius_outlier_removal(downpcd,
#            nb_points=neighbs, radius=3*voxel) ##20 se 3
#        if len(ind)/np.asarray(downpcd.points).shape[0] < 0.75:
#            neighbs -= 5
#            break
#        neighbs += 5
#    
#    cl1,ind = radius_outlier_removal(downpcd,
#            nb_points=neighbs, radius=3*voxel)   
        
    
    inlier_cloud = select_down_sample(downpcd, ind)
    inliers = np.asarray(inlier_cloud.points)
            
    X = inliers[:,0]
    Y = inliers[:,1]
    Z = inliers[:,2]
    C = np.asarray(inlier_cloud.colors)
    
    
    #
    ind = np.where((Z < 0.20) & (Z > bottom) & (X > 0) & (X < 0.5) & (Y > 0) & (Y < 0.5))
    
    X = X[ind]
    Y = Y[ind]
    Z = Z[ind]
    C = C[ind]
    
    
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
    
    #bottom = 0
    
    # Calculate volume
    vol = volume_estimation(X, Y, Z, bottom, voxel, avg_dist = 4)
    print("Volume is: ", vol)
    Vs.append(vol)
    
    end = time.time()
    print("Time elapsed: ", end - start)
    
    print("Z_placemat median: ", np.median(Z_placemat), " mean: ", np.mean(Z_placemat), " diff : ", (np.median(Z_placemat) - np.mean(Z_placemat))/np.mean(Z_placemat))


    points = np.zeros((len(X),3))
    points[:,0] = X
    points[:,1] = Y
    points[:,2] =  max(bottom, 0) 
    
    real_points = np.copy(points)
    real_points[:,2] = Z
    
    
    tri = Delaunay(points[:,0:2], furthest_site = False)
    
    delete_list = []
    V = 0
    for i in range(len(tri.simplices)):
        p = points[tri.simplices[i]]
        d1 = np.linalg.norm(p[0]- p[1])
        d2 = np.linalg.norm(p[2]- p[1])
        d3 = np.linalg.norm(p[0]- p[2])
        avg = (d1 + d2 + d3)/3
        if avg > 3*voxel:
            delete_list.append(i)
            continue
        point_set = np.vstack([points[tri.simplices[i]], real_points[tri.simplices[i]]])
        Q = ConvexHull(point_set)
        V = V + Q.volume
    V = V*1000
    
    
    
    end = time.time()
    print(end-start)
    
    
    print(V)
    
    a = np.delete(tri.simplices, delete_list, axis=0)
    
    #X = points[:,0]
    #Y= points[:,1]
    #Z = points[:,2]
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.plot_trisurf(X, Y, Z, triangles=a, cmap=plt.cm.Spectral)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()
