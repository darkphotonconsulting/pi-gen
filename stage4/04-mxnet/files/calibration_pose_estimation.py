import cv2 
import numpy as np
import glob
import math 
import time, os, sys
import argparse

n_rows = 7 
n_cols = 7
n_cols_and_rows = (n_cols, n_rows)
n_rows_and_cols = (n_rows, n_cols)


with np.load('CalibrationData/calib.npz') as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
  print("loaded mtx: \n{}".format(mtx))
  print("loaded dist: \n{}".format(dist))
  print("loaded rvecs: \n{}".format(rvecs))
  print("loaded tvecs: \n{}".format(tvecs))




def draw(img, corners, imgpts):
  corner = tuple(corners[0].ravel())
  imgpoint1  = tuple(imgpts[0].ravel())
  imgpoint2  = tuple(imgpts[1].ravel())
  imgpoint3  = tuple(imgpts[2].ravel())

  print("Image Point 1: {}".format(imgpoint1))
  print("Image Point 2: {}".format(imgpoint2))
  print("Image Point 3: {}".format(imgpoint3))
  print("using corner:\n{}".format(corner))
  #make sure the GRAY mode images are converted to BGR mode 
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  cv2.line(img, corner, imgpoint1, (255,0,0), 5)
  print("IMAGE SHAPE: {}".format(img.shape))
  cv2.line(img, corner, imgpoint2, (0,255,0), 5)
  print("IMAGE SHAPE: {}".format(img.shape))
  cv2.line(img, corner, imgpoint3, (0,0,255), 5)
  print("IMAGE SHAPE: {}".format(img.shape))
  return img

#
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((n_rows*n_cols, 3), np.float32)
objp[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2)
axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]).reshape(-1, 3)
objpoints = []
imgpoints = []

for fname in sorted(glob.glob('./tests/testCalibration/camera_precalib_frame*jpg')):
  print("pose estimation for {}".format(fname))
  img = cv2.imread(fname)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(img, n_rows_and_cols, None)


  if ret == True:
    objpoints.append(objp)
    print("located chessboard in image with corners: \n{}".format(corners))
    print("CORNERS SHAPE: {}".format(corners.shape))

    #corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    cv2.drawChessboardCorners(img, n_rows_and_cols, corners, ret)
    print("IMAGE SHAPE: {}".format(img.shape))
    #if corners2 is None:
    #  print("no subpixels, skipping")
    #  continue
    #print("Corners: {}".format(corners2))

    rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
    #cv2.drawChessboardCorners(img, n_rows_and_cols, corners, ret)

    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    print("IMAGE SHAPE: {}".format(img.shape))
    img = draw(img, corners, imgpts)
    print("IMAGE SHAPE: {}".format(img.shape))
    fname_pretty = os.path.basename(fname)
    cv2.imwrite('./tests/testCalibration/axis_projected_'+fname_pretty, img)
    print("wrote chessboard with projected axis")
    #cv2.imshow('image',img)
    #k = cv2.waitKey(0) & 0xff
    #if k == 's':
    #  cv2.imwrite('./tests/testCalibration/axis_projected_'+fname[:6]+'.png', img)
  else:
    print("chessboard corners not located in image")
