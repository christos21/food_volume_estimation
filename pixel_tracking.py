# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:56:31 2019

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
#%matplotlib auto
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

import sklearn.cluster as cl


lk_params = dict( winSize  = (15,15),
                  maxLevel = 4,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

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

with open('calibration.yaml') as f:
    intrinsic = yaml.load(f)
    
cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
distCoeffs = np.array(intrinsic.get('distCoeffs'))

vid = "/media/christos/Windows/Users/user/Desktop/samples/normal/food2.mp4"
start = time.time()
cap = cv2.VideoCapture(vid)

cameraMatrix, distCoeffs = calibrate_cap(cap, num = 10, step = 20)
#cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 10)

#cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 80)

if cameraMatrix[0,0] > 1800 or cameraMatrix[1,1] > 1800 :

    with open('calibration.yaml') as f:
        intrinsic = yaml.load(f)
        
    cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
    distCoeffs = np.array(intrinsic.get('distCoeffs'))
    
end = time.time()
print("Time for calibration: ", end - start)

start = time.time()
cap = cv2.VideoCapture(vid)
step = 10
for i in range(0*step):
    ret, img1 = cap.read()

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
ut=0


#cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frame1.shape[0:2][::-1], 0, frame1.shape[0:2][::-1])


h, w = prvs.shape  
current_transformation = np.identity(4)

u = 0

clouds = 3

while u < clouds:
    
    for t in range(80):
        ret, im1 = cap.read()
    
    if im1 is None:
        break
    
    while is_blur(im1, 100):
        ret, im1 = cap.read()
        if im1 is None:
            break
        
    frame1 = im1
#    frame1 = cv2.undistort(frame1, cameraMatrix, distCoeffs, None, cameraMatrix)

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
      
    _, rvec1, tvec1, cor1, ids1 = pose_estimation(frame1, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
    
    if rvec1 is None:
        continue
    
    a1, _ = cv2.Rodrigues(rvec1)
    L1 = np.dot(cameraMatrix, np.hstack((a1,tvec1)))
        
    mask1 = placemat(frame1, cameraMatrix, rvec1, tvec1)

    dish1, d1 = food(frame1, mask1, cor1)
    
    plt.figure()
    plt.imshow(dish1)
    
    
    im = cv2.bitwise_and(prvs, prvs, mask = dish1)
#    ret, th = cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#    edg = cv2.Canny(im, 0.1*ret, 1.2*ret)
    
    if u == 0:
        corners = cv2.goodFeaturesToTrack(im,2000,0.001,3)
        ind = corners
        rep = 20
        camera_move = 0.15
#        corners = np.int0(corners)
#    
#        for i in corners:
#            x,y = i.ravel()
#            cv2.circle(im,(x,y),3,255,-1)
#        
#        plt.imshow(im),plt.show()
    else:
        rep = 10
        corners = cv2.goodFeaturesToTrack(im,5000,0.01,3)
        ind = corners
        if corners.shape[0] < 700 :         
            camera_move = 0.05
        rows, cols = np.where(dish1 > 0)
        
        ind = np.hstack((cols.reshape(-1,1), rows.reshape(-1,1))).reshape(-1,1,2)
        ind = ind.astype("float32")
        
        dish1 = dish1.reshape(-1)
        
        if ind.shape[0] < 1:
            continue
        
        f = [randint(0, ind.shape[0]-1) for q in range(0, 5000)]
        
        ind = ind[f]

    pr = np.copy(ind)
    
    for t in range(rep):
        ret,im2 = cap.read()
        ret,im2 = cap.read()
#        ret,frame2 = cap.read()
        if im2 is None:
            break
        while is_blur(im2, 100):
            ret, im2 = cap.read()
            if im2 is None:
                break
            
        frame2 = im2
#        frame2 = cv2.undistort(frame2, cameraMatrix, distCoeffs, None, cameraMatrix)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(frame2, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        a2, _ = cv2.Rodrigues(rvec2)
        
        
        next1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next1, pr, None, **lk_params)
        
        good_new = p1[st==1]
        ind = ind[st==1].reshape(-1,1,2)
        
        prvs = next1
        pr = good_new.reshape(-1,1,2)
        
        if np.linalg.norm(np.matmul(a2.T, tvec2)[0:3]-np.matmul(a1.T, tvec1)[0:3]) > camera_move and u > 0:
#            t1 = np.arccos((np.trace(a1)-1)/2)
#            wmega1 = 1/(2*np.sin(t1))*np.array([a1[2,1]-a1[1,2], a1[0,2]-a1[2,0], a1[1,0]-a1[0,1]])
#            a2, _ = cv2.Rodrigues(rvec2)
#            t2 = np.arccos((np.trace(a2)-1)/2)
#            wmega2 = 1/(2*np.sin(t2))*np.array([a2[2,1]-a2[1,2], a2[0,2]-a2[2,0], a2[1,0]-a2[0,1]])
            print(t, np.linalg.norm(tvec1-tvec2))
#            print(-np.matmul(a1.T, tvec1))
#            print(-np.matmul(a2.T, tvec2))
#            print(np.linalg.norm(np.matmul(a2.T, tvec2)[0:3]-np.matmul(a1.T, tvec1)[0:3]))
            break
#    
#    if np.abs(t1 - t2) < 0.02:
#        continue
#    _, rvec2, tvec2, cor2, ids2 = pose_estimation(frame2, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
    if im2 is None:
        break
    if (rvec1 is None or rvec2 is None):
        continue

    a2, _ = cv2.Rodrigues(rvec2)
    L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))
    
    mask2 = placemat(frame2, cameraMatrix, rvec2, tvec2)
    dish2, d2 = food(frame2, mask2, cor2)

    pts1 = (ind.reshape(-1,2)).T
    pts2 = (good_new).T
    
    if pts1.shape[0] is 0 or pts2.shape[0] is 0:
        continue

    img3 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB) 
    color1 = [img3[int(pts1[1,t]), int(pts1[0,t])] for t in range(pts1.shape[1])]
    color1 = np.array(color1)

    points3d = cv2.triangulatePoints(L1, L2, pts1[:2,...], pts2[:2,...]) 
    pts3D = points3d/points3d[3]
    
    pts3D = pts3D.T
    pts1 = pts1.T
    pts2 = pts2.T
    
    reprojected1, _ = cv2.projectPoints(pts3D[:,0:3], rvec1, tvec1, cameraMatrix, distCoeffs)
    reprojected2, _ = cv2.projectPoints(pts3D[:,0:3], rvec2, tvec2, cameraMatrix, distCoeffs)
     
    delete_list = []
    er = []
    
    for j in range(len(pts1)):
        er1 = np.linalg.norm(reprojected1[j]- pts1[j])
        er2 = np.linalg.norm(reprojected2[j]- pts2[j])
        er.append( (er1+er2)/2)
        if er1 > 4 or er2 > 4:
            delete_list.append(j)
    
    
    pts3D = np.delete(pts3D, delete_list, axis=0)
    
    pts1 = np.delete(pts1, delete_list, axis=0)
    pts2 = np.delete(pts2, delete_list, axis=0)
 
    color1 = np.delete(color1, delete_list, axis=0)
    er = np.delete(er, delete_list, axis = 0)
    
#    if u > 0:
#        er_ind = np.argsort(er)
#        
#        m = min(3000, len(er))
#        er_ind = er_ind[0:m]
#        
#        pts1 = pts1[er_ind,:]
#        pts2 = pts2[er_ind,:]
#        color1 = color1[er_ind,:]
#        pts3D = pts3D[er_ind,:]


    if pts1.shape[0] < 10:
        continue
    
    
#    _, rvecs, tvecs, inliers1  = cv2.solvePnPRansac(pts3D[:,0:3], pts1, cameraMatrix, distCoeffs,
#                rvec = rvec1, tvec = tvec1, useExtrinsicGuess = 1, flags = 0)
#    _, rvecs2, tvecs2, inliers2  = cv2.solvePnPRansac(pts3D[:,0:3], pts2, cameraMatrix, distCoeffs,
#                rvec = rvec2, tvec = tvec2, useExtrinsicGuess = 1, flags = 0)
#    
#    idd = []
#    
#    if inliers1 is None or inliers2 is None:
#        continue
    
    
    ut += 1
#    
#    for g in inliers1:
#        if g in inliers2:
#            idd.append(g)
#            
#    newpts3d = pts3D[idd,:]
#    pts3D = newpts3d.reshape(-1,4)
#    color1 = color1[idd,:].reshape(-1,3)
    
    u = u + 1
    
    A = pts3D[:,0]
    B = pts3D[:,1]
    C = pts3D[:,2]
    
   
    ind = np.where((C < 0.20) & (C > 0.0) & (A > 0) & (A < 0.5) & (B > 0) & (B < 0.5))

    X = A[ind]
    Y = B[ind]
    Z = C[ind]
    color1 = color1[ind].reshape(-1,3)
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    scat = ax.scatter(X, Y, Z, c = color1/255.0, s = 3)
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()


    xyz = np.vstack([X,Y,Z]).transpose()
    
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(color1/255.0)
    
    img_aruco = np.copy(frame1)
    img_aruco = aruco.drawAxis(img_aruco, cameraMatrix, distCoeffs, rvec1, tvec1, 0.1)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB))
    img_aruco = np.copy(frame2)
    img_aruco = aruco.drawAxis(img_aruco, cameraMatrix, distCoeffs, rvec2, tvec2, 0.1)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB))    
    

    if ut==1:
        final = pcd
        Xs = X
        Ys = Y
        Zs = Z
        color = color1
    else:
        tr = registration_icp(pcd, final, 0.01, current_transformation,
            TransformationEstimationPointToPoint())#,
#            ICPConvergenceCriteria(max_iteration = 2000))
        pcd.transform(tr.transformation)
        t = np.asarray(final.points)
        col = np.asarray(final.colors)
        s = np.asarray(pcd.points)
        xyz = np.vstack([t,s])
        final = PointCloud()
        final.points = Vector3dVector(xyz)
        final.colors = Vector3dVector(np.vstack([col, color1/255.0]))
        
#        current_transformation = final.transformation
        Xs = np.hstack([Xs, X])
        Ys = np.hstack([Ys, Y])
        Zs = np.hstack([Zs, Z])
        color = np.vstack([color, color1])
        
  



xyz = np.asarray(final.points)
Xs = xyz[:,0]
Ys = xyz[:,1]
Zs = xyz[:,2]
C = np.asarray(final.colors)

bottom = 0.009
ind = np.where((Zs < 0.20) & (Zs > bottom) & (Xs > 0) & (Xs < 0.5) & (Ys > 0) & (Ys < 0.5))

X = Xs[ind]
Y = Ys[ind]
Z = Zs[ind]
C = color[ind]



xyz = np.vstack([X,Y,Z]).transpose()



###DB scan
#clustering = cl.DBSCAN(eps=0.007, min_samples=30, metric='euclidean',
#                       metric_params=None, algorithm='kd_tree', leaf_size=10, p=None, n_jobs=None).fit(xyz)
#
#id1 = np.where(clustering.labels_==0)
#
#X = xyz[id1,0].reshape(-1)
#Y = xyz[id1,1].reshape(-1)
#Z = xyz[id1,2].reshape(-1)
#C = C[id1,:].reshape(-1,3)
#
#xyz = np.vstack([X,Y,Z]).transpose()



pcd = PointCloud()
pcd.points = Vector3dVector(xyz)
pcd.colors = Vector3dVector(C)

voxel = 0.005
downpcd = voxel_down_sample(pcd, voxel_size = voxel)

neighbs = 20
while True:
    cl1,ind = radius_outlier_removal(downpcd,
        nb_points=neighbs, radius=3*voxel) ##20 se 3
    if len(ind)/np.asarray(downpcd.points).shape[0] < 0.75:
        neighbs -= 5
        break
    neighbs += 5

cl1,ind = radius_outlier_removal(downpcd,
        nb_points=neighbs, radius=3*voxel)   
    
#    
#cl1,ind = radius_outlier_removal(downpcd,
#        nb_points=40, radius=3*voxel) ##20 se 3
#    
#
#if len(ind)/np.asarray(downpcd.points).shape[0] < 0.75:
#    cl1,ind = radius_outlier_removal(downpcd,
#        nb_points=20, radius=3*voxel)
#    
    
inlier_cloud = select_down_sample(downpcd, ind)
inliers = np.asarray(inlier_cloud.points)
        
        
X = inliers[:,0]
Y = inliers[:,1]
Z = inliers[:,2]
C = np.asarray(inlier_cloud.colors)



#X = np.asarray(downpcd.points)[:,0]
#Y = np.asarray(downpcd.points)[:,1]
#Z = np.asarray(downpcd.points)[:,2]
#C = np.asarray(downpcd.colors)



fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
scat = ax.scatter(X, Y, Z, c = C/255.0, s = 3)
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
plt.show()


#bottom = 0

V = volume_estimation(X, Y, Z, bottom, voxel, avg_dist = 3) #na balw real_points opws katw
print("Volume is: ", V)

end = time.time()
print("Time for volume: ", end - start)

#
##
#
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
