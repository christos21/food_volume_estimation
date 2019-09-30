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

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

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

# Load intrinsic parameters
with open('calibration.yaml') as f:
    intrinsic = yaml.load(f)
    
cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
distCoeffs = np.array(intrinsic.get('distCoeffs'))

# video path
vid = "/home/christos/Downloads/n11.mp4"

# start timing for calibration
start = time.time()
cap = cv2.VideoCapture(vid)

# calibrate camera
cameraMatrix, distCoeffs = calibrate_cap(cap, num=5, step=40)
#cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 10)
#cameraMatrix, distCoeffs = calibrate_cap1(cap, step = 80)


# if calibration is way off, just load the precalculated parameters (for example food7)
if cameraMatrix[0, 0] > 1800 or cameraMatrix[1, 1] > 1800:
    with open('calibration.yaml') as f:
        intrinsic = yaml.load(f)
    cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
    distCoeffs = np.array(intrinsic.get('distCoeffs'))

# stop timing for calibration
end = time.time()
print("Time for calibration: ", end - start)

# start timing for volume estimation
start = time.time()
cap = cv2.VideoCapture(vid)

# skip some frames (probably user won't move the camera from the beginning)
step = 10
for i in range(2*step):
    ret, img1 = cap.read()

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
ut = 0

#cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, frame1.shape[0:2][::-1], 0, frame1.shape[0:2][::-1])

# get images shape (currently 1920x1080)
h, w = prvs.shape
# initialization of transformation matrix used for ICP
current_transformation = np.identity(4)

u = 0
clouds = 3

# repeat the 3d reconstruction with different frames
while u < clouds:

    # skip some frames so that the different point clouds won't be from similar points of view
    for t in range(30):
        ret, im1 = cap.read()

    # if video is finished break
    if im1 is None:
        break

    # check if image is blur and keep one that is not
    while is_blur(im1, 100):
        ret, im1 = cap.read()
        if im1 is None:
            break

    # this is the first frame of the sequence
    frame1 = im1

    # RGB to grayscale
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # pose estimation
    _, rvec1, tvec1, cor1, ids1 = pose_estimation(frame1, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)

    # if can't estimate pose (eg there is not a single aruco marker in image) continue
    if rvec1 is None:
        continue

    # estimate rotation matrix
    a1, _ = cv2.Rodrigues(rvec1)
    # estimate projection matrix as cameraMatrix*[R | t]
    L1 = np.dot(cameraMatrix, np.hstack((a1, tvec1)))

    # Get placemat mask
    mask1 = placemat(frame1, cameraMatrix, rvec1, tvec1)

    # get food and dish mask
    dish1, d1 = food(frame1, mask1, cor1)

    # # plot food mask
    # plt.figure()
    # plt.imshow(dish1)

    # keep only food pixels in the grayscale image
    im = cv2.bitwise_and(prvs, prvs, mask=dish1)

    # if this is the first point cloud use goodFeaturesToTack in order to have
    # the strongest feature points
    if u == 0:
        corners = cv2.goodFeaturesToTrack(im, 1000, 0.001, 10)
        ind = corners
        rep = 30
        camera_move = 0.2

    # if it's not the first point cloud we find random food pixels for tracking in order to
    # reconstruct foods with no strong feature points
    else:
        rep = 10
        camera_move = 0.2
        # corners = cv2.goodFeaturesToTrack(im, 5000, 0.01, 3)
        # ind = corners
        # if corners.shape[0] < 700:
        #     camera_move = 0.05

        # get food pixels
        rows, cols = np.where(dish1 > 0)

        # array with food pixels
        ind = np.hstack((cols.reshape(-1, 1), rows.reshape(-1, 1))).reshape(-1, 1, 2)
        ind = ind.astype("float32")

        dish1 = dish1.reshape(-1)

        # if there are no food pixels continue
        if ind.shape[0] < 1:
            continue

        # keep random 5000 food pixels
        f = [randint(0, ind.shape[0]-1) for q in range(0, 3000)]
        ind = ind[f]

    pr = np.copy(ind)

    # Here happens the pixel tracking process.
    # The tracking can't continue for more than rep frames.
    for t in range(rep):
        ret, im2 = cap.read()
        ret, im2 = cap.read()
        # if video is finished break
        if im2 is None:
            break
        # find a non blur frame
        while is_blur(im2, 100):
            ret, im2 = cap.read()
            if im2 is None:
                break

        frame2 = im2

        # estimate the pose and find rotation matrix
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(frame2, board, aruco_dict, arucoParams, cameraMatrix, distCoeffs)
        a2, _ = cv2.Rodrigues(rvec2)

        next1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # calculate optical flow aka track the food pixels
        p1, st, err = cv2.calcOpticalFlowPyrLK(prvs, next1, pr, None, **lk_params)

        # keep only the good tracked pixels
        good_new = p1[st == 1]
        ind = ind[st == 1].reshape(-1, 1, 2)

        # renew the frame and the pixels that will be tracked
        prvs = next1
        pr = good_new.reshape(-1, 1, 2)

        # if camera has moved enough during this tracking, stop tracking
        if np.linalg.norm(np.matmul(a2.T, tvec2)[0:3]-np.matmul(a1.T, tvec1)[0:3]) > camera_move and u > 0:
            print(t, np.linalg.norm(tvec1-tvec2))
            break

    # if video is finished break
    if im2 is None:
        break

    # if can't estimate pose continue
    if (rvec1 is None or rvec2 is None):
        continue

    # find rotation and projection matrix of the end frame
    a2, _ = cv2.Rodrigues(rvec2)
    L2 = np.dot(cameraMatrix, np.hstack((a2,tvec2)))

    # find placemat, food and dish masks
    mask2 = placemat(frame2, cameraMatrix, rvec2, tvec2)
    dish2, d2 = food(frame2, mask2, cor2)

    # pts1 are the food pixels of the start frame and pts2 the corresponding pixels of the end frame
    pts1 = (ind.reshape(-1, 2)).T
    pts2 = (good_new).T

    # if there is no corresponding frames between start and end frames continue
    if pts1.shape[0] is 0 or pts2.shape[0] is 0:
        continue

    # keep colors of these pixels form the start frame
    # (just for visualization in the point cloud)
    img3 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    color1 = [img3[int(pts1[1, t]), int(pts1[0, t])] for t in range(pts1.shape[1])]
    color1 = np.array(color1)

    # triangulate to find the real 3d points
    points3d = cv2.triangulatePoints(L1, L2, pts1[:2, ...], pts2[:2, ...])
    pts3D = points3d/points3d[3]

    pts3D = pts3D.T
    pts1 = pts1.T
    pts2 = pts2.T

    # remove points with high reprojection error
    reprojected1, _ = cv2.projectPoints(pts3D[:,0:3], rvec1, tvec1, cameraMatrix, distCoeffs)
    reprojected2, _ = cv2.projectPoints(pts3D[:,0:3], rvec2, tvec2, cameraMatrix, distCoeffs)

    delete_list = []
    er = []

    for j in range(len(pts1)):
        er1 = np.linalg.norm(reprojected1[j] - pts1[j])
        er2 = np.linalg.norm(reprojected2[j] - pts2[j])
        er.append((er1+er2)/2)
        if er1 > 5 or er2 > 5:
            delete_list.append(j)

    pts3D = np.delete(pts3D, delete_list, axis=0)
    pts1 = np.delete(pts1, delete_list, axis=0)
    pts2 = np.delete(pts2, delete_list, axis=0)

    color1 = np.delete(color1, delete_list, axis=0)
    er = np.delete(er, delete_list, axis = 0)

    # if the remaining points are < 10 continue
    if pts1.shape[0] < 10:
        continue

    #RANSAC missing

    ut += 1
    u += 1

    # from the reconstructed points remove those that are outside a 0.5mx0.5mx0.2m box
    # WCC (0,0,0) is a placemat point so these points are outside the placemat (and food can't be taller than 0.2m)
    A = pts3D[:, 0]
    B = pts3D[:, 1]
    C = pts3D[:, 2]

    ind = np.where((C < 0.20) & (C > 0.0) & (A > 0) & (A < 0.5) & (B > 0) & (B < 0.5))

    X = A[ind]
    Y = B[ind]
    Z = C[ind]
    color1 = color1[ind].reshape(-1,3)

    # # Plot the reconstructed points
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # scat = ax.scatter(X, Y, Z, c = color1/255.0, s = 3)
    # max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    # mid_x = (X.max()+X.min()) * 0.5
    # mid_y = (Y.max()+Y.min()) * 0.5
    # mid_z = (Z.max()+Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # plt.show()

    xyz = np.vstack([X, Y, Z]).transpose()

    # create the point cloud
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(color1/255.0)

    # # plot start and end frame with WCS axis
    # img_aruco = np.copy(frame1)
    # img_aruco = aruco.drawAxis(img_aruco, cameraMatrix, distCoeffs, rvec1, tvec1, 0.1)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB))
    # img_aruco = np.copy(frame2)
    # img_aruco = aruco.drawAxis(img_aruco, cameraMatrix, distCoeffs, rvec2, tvec2, 0.1)
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_aruco, cv2.COLOR_BGR2RGB))

    # if this is the first iteration just initialize the final point cloud
    if ut == 1:
        final = pcd
        Xs = X
        Ys = Y
        Zs = Z
        color = color1

    # use ICP to align the current point cloud to the final
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


# final coordinates and color
xyz = np.asarray(final.points)
Xs = xyz[:, 0]
Ys = xyz[:, 1]
Zs = xyz[:, 2]
C = np.asarray(final.colors)

# bottom refers to the dish height. We use a 9mm height. Basically food points must have Z > bottom
bottom = 0.009
# remove points that are below dish height.
ind = np.where((Zs < 0.20) & (Zs > bottom) & (Xs > 0) & (Xs < 0.5) & (Ys > 0) & (Ys < 0.5))

X = Xs[ind]
Y = Ys[ind]
Z = Zs[ind]
C = C[ind]

xyz = np.vstack([X, Y, Z]).transpose()

pcd = PointCloud()
pcd.points = Vector3dVector(xyz)
pcd.colors = Vector3dVector(C)

#Downsample the points to a specific voxel size. 1 pixel/voxel
voxel = 0.005
downpcd = voxel_down_sample(pcd, voxel_size = voxel)

# Remove points without many neighbors.
# Number of neighbors is a multiplicant of 5 but not specific.
# It is the highest one that won't remove more than 75% of the total points.
neighbs = 20
while True:
    cl1, ind = radius_outlier_removal(downpcd,
        nb_points=neighbs, radius=3*voxel)
    if len(ind)/np.asarray(downpcd.points).shape[0] < 0.75:
        neighbs -= 5
        break
    neighbs += 5

cl1, ind = radius_outlier_removal(downpcd,
        nb_points=neighbs, radius=3*voxel)   

inlier_cloud = select_down_sample(downpcd, ind)
inliers = np.asarray(inlier_cloud.points)

X = inliers[:, 0]
Y = inliers[:, 1]
Z = inliers[:, 2]
C = np.asarray(inlier_cloud.colors)

# # plot the final point cloud
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_aspect('equal')
# scat = ax.scatter(X, Y, Z, c = C/255.0, s = 3)
# max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
# mid_x = (X.max()+X.min()) * 0.5
# mid_y = (Y.max()+Y.min()) * 0.5
# mid_z = (Z.max()+Z.min()) * 0.5
# ax.set_xlim(mid_x - max_range, mid_x + max_range)
# ax.set_ylim(mid_y - max_range, mid_y + max_range)
# ax.set_zlim(mid_z - max_range, mid_z + max_range)
# plt.show()

# estimate the volume
V = volume_estimation(X, Y, Z, bottom, voxel, avg_dist=3)
print("Volume is: ", V)

# stop timing
end = time.time()
print("Time for volume: ", end - start)