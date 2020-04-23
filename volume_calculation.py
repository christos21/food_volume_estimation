import cv2
from open3d import *
from cv2 import aruco
import yaml
import numpy as np
from auxiliary_functions import *
import time
from random import randint
from matplotlib import pyplot as plt


def volume_with_pixel_tracking(vid, calibration_file='calibration.yaml'):

    # LK tracking algorithm parameters
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

    # start timing for calibration
    start = time.time()
    # capture video
    cap = cv2.VideoCapture(vid)

    # calibrate camera
    cameraMatrix, distCoeffs = calibrate_cap(cap, num=5, step=40)

    # if calibration is way off, just load the pre-calculated parameters
    if cameraMatrix[0, 0] > 1800 or cameraMatrix[1, 1] > 1800:
        with open(calibration_file) as f:
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
    for i in range(2 * step):
        cap.read()

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

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

            # get food pixels
            rows, cols = np.where(dish1 > 0)

            # array with food pixels
            ind = np.hstack((cols.reshape(-1, 1), rows.reshape(-1, 1))).reshape(-1, 1, 2)
            ind = ind.astype("float32")

            # if there are no food pixels continue
            if ind.shape[0] < 1:
                continue

            # keep random 5000 food pixels
            f = [randint(0, ind.shape[0] - 1) for q in range(0, 3000)]
            ind = ind[f]

        pr = np.copy(ind)

        # Here starts the pixel tracking process.
        # The tracking can't continue for more than rep frames.
        for t in range(rep):
            cap.read()
            ret, im2 = cap.read()
            # if video is finished, break
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
            if np.linalg.norm(np.matmul(a2.T, tvec2)[0:3] - np.matmul(a1.T, tvec1)[0:3]) > camera_move and u > 0:
                print(t, np.linalg.norm(tvec1 - tvec2))
                break

        # if video is finished break
        if im2 is None:
            break

        # if can't estimate pose continue
        if rvec1 is None or rvec2 is None:
            continue

        # find rotation and projection matrix of the end frame
        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2, tvec2)))

        # pts1 are the food pixels of the start frame and pts2 the corresponding pixels of the end frame
        pts1 = (ind.reshape(-1, 2)).T
        pts2 = good_new.T

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
        pts3D = points3d / points3d[3]

        pts3D = pts3D.T
        pts1 = pts1.T
        pts2 = pts2.T

        # remove points with high reprojection error
        reprojected1, _ = cv2.projectPoints(pts3D[:, 0:3], rvec1, tvec1, cameraMatrix, distCoeffs)
        reprojected2, _ = cv2.projectPoints(pts3D[:, 0:3], rvec2, tvec2, cameraMatrix, distCoeffs)

        delete_list = []
        er = []

        for j in range(len(pts1)):
            er1 = np.linalg.norm(reprojected1[j] - pts1[j])
            er2 = np.linalg.norm(reprojected2[j] - pts2[j])
            er.append((er1 + er2) / 2)
            if er1 > 5 or er2 > 5:
                delete_list.append(j)

        pts3D = np.delete(pts3D, delete_list, axis=0)
        pts1 = np.delete(pts1, delete_list, axis=0)
        pts2 = np.delete(pts2, delete_list, axis=0)

        color1 = np.delete(color1, delete_list, axis=0)
        er = np.delete(er, delete_list, axis=0)

        # if the remaining points are < 10 continue
        if pts1.shape[0] < 10:
            continue

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
        color1 = color1[ind].reshape(-1, 3)

        xyz = np.vstack([X, Y, Z]).transpose()

        # create the point cloud
        pcd = PointCloud()
        pcd.points = Vector3dVector(xyz)
        pcd.colors = Vector3dVector(color1 / 255.0)

        # if this is the first iteration just initialize the final point cloud
        if u == 1:
            final = pcd

        # use ICP to align the current point cloud to the final
        else:
            tr = registration_icp(pcd, final, 0.01, current_transformation,
                                  TransformationEstimationPointToPoint())
            pcd.transform(tr.transformation)
            t = np.asarray(final.points)
            col = np.asarray(final.colors)
            s = np.asarray(pcd.points)
            xyz = np.vstack([t, s])
            final = PointCloud()
            final.points = Vector3dVector(xyz)
            final.colors = Vector3dVector(np.vstack([col, color1 / 255.0]))

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

    # Down sample the points to a specific voxel size. 1 pixel/voxel
    voxel = 0.005
    downpcd = voxel_down_sample(pcd, voxel_size=voxel)

    # Remove points without many neighbors.
    # Number of neighbors is a multiplicant of 5 but not specific.
    # It is the highest one that won't remove more than 75% of the total points.
    neighbs = 20
    while True:
        cl1, ind = radius_outlier_removal(downpcd,
                                          nb_points=neighbs, radius=3 * voxel)
        if len(ind) / np.asarray(downpcd.points).shape[0] < 0.75:
            neighbs -= 5
            break
        neighbs += 5

    cl1, ind = radius_outlier_removal(downpcd,
                                      nb_points=neighbs, radius=3 * voxel)

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

    return V


def volume_with_stereo_matching(vid, calibration_file='calibration.yaml'):

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

    markers = a[0] * a[1]
    full = (a[1] - 1) * (b + c) + b
    real = []
    for i in range(markers):
        x = i % a[0]
        y = i // a[0]
        real.append([[x * (b + c), full - y * (b + c), 0],
                     [x * (b + c) + b, full - y * (b + c), 0],
                     [x * (b + c) + b, full - y * (b + c) - b, 0],
                     [x * (b + c), full - y * (b + c) - b, 0]])

    # start timing for calibration
    start = time.time()
    cap = cv2.VideoCapture(vid)
    # calibrate
    cameraMatrix, distCoeffs = calibrate_cap(cap, num=5, step=70)

    # if calibration is wrong, load precalculated intrinsic
    if cameraMatrix[0, 0] > 1800 or cameraMatrix[1, 1] > 1800:
        # Load pre calculated intrinsic parameters
        with open(calibration_file) as f:
            intrinsic = yaml.load(f)
        cameraMatrix = np.array(intrinsic.get('cameraMatrix'))
        distCoeffs = np.array(intrinsic.get('distCoeffs'))

    # stop timing for calibartion
    end = time.time()
    print('Time for calibration: ', end-start)

    # start timing for volume estimation
    start = time.time()
    cap = cv2.VideoCapture(vid)
    step = 10
    # skip some frames
    for i in range(2 * step):
        ret, img1 = cap.read()

    img = []
    img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

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
        if ut + half >= len(img):
            br = True
            break
        img2 = img[ut + half]

        # Find extrinsic parameters
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(img1, board, aruco_dict, arucoParams, cameraMatrix,
                                                      distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(img2, board, aruco_dict, arucoParams, cameraMatrix,
                                                      distCoeffs)

        # find rotation and projection matrices
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1, tvec1)))

        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2, tvec2)))

        # relative R and T between the two frames
        R = np.dot(a2, np.linalg.inv(a1))
        T = - np.dot(R, tvec1) + tvec2

        # find placemat masks
        mask1 = placemat(img1, cameraMatrix, rvec1, tvec1)
        mask2 = placemat(img2, cameraMatrix, rvec2, tvec2)

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=15)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8), iterations=15)

        # keep only the placemats from the two frames
        img1, img2 = cv2.bitwise_and(img1, img1, mask=mask1), cv2.bitwise_and(img2, img2, mask=mask2)

        # frames' size
        h, w = img1.shape[:2]

        # apply rectification to the two frames
        RL, RR, PL, PR, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix, distCoeffs,
                                                          cameraMatrix, distCoeffs, (w, h), R, T,
                                                          alpha=-1)

        # If they rotated, continue and take the next pair with higher interval between the frames
        if np.abs(RL[0, 0]) < 0.95:
            half += 1
            continue

        mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RL, PL, (w, h), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, RR, PR, (w, h), cv2.CV_32FC1)

        undistorted_rectifiedL = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR)
        undistorted_rectifiedR = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR)

        # take the actual rectified images
        imgL, imgR = undistorted_rectifiedL, undistorted_rectifiedR

        # pose estimation
        _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix,
                                                      distCoeffs)
        _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix,
                                                      distCoeffs)

        # during rectification the images may be mirrored, thus the aruco markers can't be detected
        # if that's the case just flip the images
        if rvec1 is None:
            imgL = cv2.flip(imgL, 1)
            _, rvec1, tvec1, cor1, ids1 = pose_estimation(imgL, board, aruco_dict, arucoParams, cameraMatrix,
                                                          distCoeffs)

        if rvec2 is None:
            imgR = cv2.flip(imgR, 1)
            _, rvec2, tvec2, cor2, ids2 = pose_estimation(imgR, board, aruco_dict, arucoParams, cameraMatrix,
                                                          distCoeffs)

        # Find maximum horizontal translation based on the known markers' corners
        fl = False
        dist = []
        for k in range(len(ids1)):
            idd = ids1[k][0]
            if idd not in ids2:
                continue
            idd2 = np.where(ids2 == idd)[0][0]
            sss = np.abs(cor1[k][0, :, 0] - cor2[idd2][0, :, 0]).max()
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
        n = np.ceil(sss / 16) + 1

        # Initialize SGBM
        window_size = 7
        min_disp = -1
        num_disp = int(n) * 16
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=10 * 3 * window_size ** 2,
            P2=35 * 3 * window_size ** 2,
            disp12MaxDiff=-1,
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

        if np.abs(RL[2, 0]) > 0.15:
            ut += 1
            continue

        # find rotation and projection matrices of the rectified images
        a1, _ = cv2.Rodrigues(rvec1)
        L1 = np.dot(cameraMatrix, np.hstack((a1, tvec1)))

        a2, _ = cv2.Rodrigues(rvec2)
        L2 = np.dot(cameraMatrix, np.hstack((a2, tvec2)))

        R1 = np.dot(a2, np.linalg.inv(a1))
        T1 = - np.dot(R1, tvec1) + tvec2

        # find placemat, food and dish masks
        mask1 = placemat(imgL, cameraMatrix, rvec1, tvec1)
        dish1, d1 = food(imgL, mask1, cor1)

        mask2 = placemat(imgR, cameraMatrix, rvec2, tvec2)
        dish2, d2 = food(imgR, mask2, cor2)

        only_placemat1 = cv2.bitwise_and(mask1, mask1, mask=cv2.bitwise_not(d1))
        only_placemat2 = cv2.bitwise_and(mask2, mask2, mask=cv2.bitwise_not(d2))

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
                h1 += 1
                continue
            idd2 = np.where(ids2 == idd)[0][0]
            sss = np.abs(cor1[h1][0, :, 0] - cor2[idd2][0, :, 0])

            c1 = np.array(np.round(cor1[h1].reshape(4, 2)), dtype=int)

            if np.where(c1[:, 1] >= h)[0].shape[0] > 0 or np.where(c1[:, 0] >= w)[0].shape[0] > 0:
                h1 += 1
                continue

            d1 = disparity[c1[0, 1], c1[0, 0]]
            d2 = disparity[c1[1, 1], c1[1, 0]]
            d3 = disparity[c1[2, 1], c1[2, 0]]
            d4 = disparity[c1[3, 1], c1[3, 0]]

            d1 = sss[0] / d1
            d2 = sss[1] / d2
            d3 = sss[2] / d3
            d4 = sss[3] / d4

            if d1 < np.inf:
                dmod.append(d1)
            if d2 < np.inf:
                dmod.append(d2)
            if d3 < np.inf:
                dmod.append(d3)
            if d4 < np.inf:
                dmod.append(d4)

            h1 += 1

        dxxx = np.round([1000 * t for t in dmod]) / 1000
        (values, counts) = np.unique(dxxx, return_counts=True)
        ind = np.argmax(counts)
        s = values[ind]

        # get disparity for food pixels
        d = disparity[dish1 > 0]

        # some heuristics regarding the quality of the disparity map
        if np.unique(d).shape[0] > 120:
            half = half - 2
            continue
        elif np.unique(d).shape[0] < 10:  # 15
            ut += 1
            continue

        d = d * s
        # food pixels
        rows, cols = np.where(dish1 > 0)

        # get reconstructed points in camera 1 coordinate system
        with np.errstate(divide='ignore', invalid='ignore'):
            pw = 1 / (d * Q[3, 2] + Q[3, 3])

        X = cols * pw
        Y = rows * pw
        Z = Q[2, 3] * pw

        # get colors of these points just for visualization
        only_plate = cv2.bitwise_and(imgL, imgL, mask=dish1)
        colors = cv2.cvtColor(only_plate, cv2.COLOR_BGR2RGB)
        c = colors[np.where(dish1 > 0)]

        # get rid of some points that are at infitiy because they had 0 in disparity map
        ind = np.where((Z < np.inf) & (Z > -np.inf) & (X > -np.inf) & (X < np.inf) & (Y > -np.inf) & (Y < np.inf))
        X = X[ind]
        Y = Y[ind]
        Z = Z[ind]
        c = c[ind]

        # choose 5000 random points
        f = [randint(0, len(X) - 1) for q in range(0, 5000)]
        X = X[f]
        Y = Y[f]
        Z = Z[f]
        c = c[f]

        #  go from camera1 coordinate system to WCS
        p = np.array([X, Y, Z])
        pw = np.zeros((3, len(X)))
        a = np.linalg.inv(a1)
        b = np.dot(np.linalg.inv(a1), tvec1)
        for k in range(len(X)):
            pw[:, k] = np.diag(np.dot(a, p[:, k]) - b)
        X = pw[0, :]
        Y = pw[1, :]
        Z = pw[2, :]

        C = c / 255.0

        # find placemat coordintes
        d = disparity[only_placemat1 > 0]
        rows, cols = np.where(only_placemat1 > 0)
        d = d * s
        with np.errstate(divide='ignore', invalid='ignore'):
            pw = 1 / (d * Q[3, 2] + Q[3, 3])

        X_placemat = cols * pw
        Y_placemat = rows * pw
        Z_placemat = Q[2, 3] * pw

        ind = np.where((Z_placemat < np.inf) & (Z_placemat > -np.inf) &
                       (X_placemat > -np.inf) & (X_placemat < np.inf) &
                       (Y_placemat > -np.inf) & (Y_placemat < np.inf))
        X_placemat = X_placemat[ind]
        Y_placemat = Y_placemat[ind]
        Z_placemat = Z_placemat[ind]

        f = [randint(0, len(X_placemat) - 1) for q in range(0, 5000)]
        X_placemat = X_placemat[f]
        Y_placemat = Y_placemat[f]
        Z_placemat = Z_placemat[f]

        #  placemate coords from camera 1 to  WCS
        p = np.array([X_placemat, Y_placemat, Z_placemat])
        pw = np.zeros((3, len(X_placemat)))
        a = np.linalg.inv(a1)
        b = np.dot(np.linalg.inv(a1), tvec1)
        for k in range(len(X_placemat)):
            pw[:, k] = np.diag(np.dot(a, p[:, k]) - b)
        X_placemat = pw[0, :]
        Y_placemat = pw[1, :]
        Z_placemat = pw[2, :]

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
    bottom = 0.009

    # keep points that are higher than dish height
    ind = np.where((Z < 0.20) & (Z > bottom) & (X > 0) & (X < 0.5) & (Y > 0) & (Y < 0.5))
    X = X[ind]
    Y = Y[ind]
    Z = Z[ind]
    C = C[ind]

    # create point cloud
    xyz = np.vstack([X, Y, Z]).transpose()
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(C)

    # Downsample to voxel
    voxel = 0.005
    downpcd = voxel_down_sample(pcd, voxel_size=voxel)

    # Outlier removal
    cl1, ind = radius_outlier_removal(downpcd,
                                      nb_points=15, radius=3 * voxel)

    inlier_cloud = select_down_sample(downpcd, ind)
    inliers = np.asarray(inlier_cloud.points)

    X = inliers[:, 0]
    Y = inliers[:, 1]
    Z = inliers[:, 2]
    C = np.asarray(inlier_cloud.colors)

    # # Plot point cloud
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')
    # scat = ax.scatter(X, Y, Z, c=C, s=3)
    # max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    # mid_x = (X.max() + X.min()) * 0.5
    # mid_y = (Y.max() + Y.min()) * 0.5
    # mid_z = (Z.max() + Z.min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # plt.show()

    # Calculate volume
    vol = int(1000 * volume_estimation(X, Y, Z, bottom, voxel, avg_dist=4))
    print("Volume is: ", vol)

    if br:
        vol = 0

    # stop timing for volume estimation
    end = time.time()
    print("Time elapsed: ", end - start)

    return vol