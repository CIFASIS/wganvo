#
# This file is part of wganvo.
#
# Copyright (C) 2019 Javier Cremona (CIFASIS-CONICET)
# For more information see <https://github.com/CIFASIS/wganvo>
#
# wganvo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# wganvo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with wganvo. If not, see <http://www.gnu.org/licenses/>.
#

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
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.knnMatch(des1, des2, k=1)
    # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []

    for match in matches:
        if match:
            # print('%d -> %d: %f' % (match[0].queryIdx, match[0].trainIdx, match[0].distance))
            m = match[0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=mask.ravel().tolist(),
    #                    flags=0)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    # plt.imshow(img3), plt.show()

    pts1 = pts1.transpose().astype(np.float32)
    pts2 = pts2.transpose().astype(np.float32)

    same_line_mask = pts1[1, :] == pts2[1, :]
    # threshold =  (pts1[1, :] + 1) == pts2[1, :]
    # threshold |= (pts2[1, :] + 1) == pts1[1, :]
    # same_line_mask |= threshold
    pts1 = pts1[:, same_line_mask]
    pts2 = pts2[:, same_line_mask]

    return pts1, pts2


# Input: P1 -> projection matrix
#        P2 -> projection matrix
#        x1 -> 2xN array of points
#        x2 -> 2xN array of points
# Output: X -> 4xN array of 3D points (homogeneous coordinates)
def triangulatePoints(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1, P2, x1, x2)
    return X / X[3]

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
