import cv2
import numpy as np
from matplotlib import pyplot as plt

# Input: stereo images
# Output: pts1 -> 2xN array of points from left image
#         pts2 -> 2xN array of points from right image
def matcher(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # Brute Force Matcher parameters
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    pts1 = pts1.transpose().astype(np.float32)
    pts2 = pts2.transpose().astype(np.float32)
    return pts1, pts2


# Input: P1 -> projection matrix
#        P2 -> projection matrix
#        x1 -> 2xN array of points
#        x2 -> 2xN array of points
# Output: X -> 4xN array of 3D points (homogeneous coordinates)
def triangulatePoints(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1, P2, x1, x2)
    return X/X[3]



# folder = '/home/jcremona/data/03/'
# filename = '000000.png'
#
# img1 = cv2.imread(folder + 'image_0/' + filename,0)  #queryimage # left image
# img2 = cv2.imread(folder + 'image_1/' + filename,0) #trainimage # right image
#
# # KITTI Seq 3 left camera calibration
# K0 =  np.matrix([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 0.000000000000e+00],
#                  [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
#                  [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
#
# # KITTI Seq 3 right camera calibration
# K1 = np.matrix([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.875744000000e+02],
#                 [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 0.000000000000e+00],
#                 [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
#
# # KITTI Seq 3 1st frame pose
# Rt0 = np.matrix([[1.000000e+00, -1.822835e-10, 5.241111e-10, -5.551115e-17],
#                  [-1.822835e-10, 9.999999e-01, -5.072855e-10, -3.330669e-16],
#                  [5.241111e-10, -5.072855e-10, 9.999999e-01, 2.220446e-16],
#                  [0.,0.,0.,1.]])
