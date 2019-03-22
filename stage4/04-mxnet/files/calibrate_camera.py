import os, sys, glob, argparse
import numpy as np 
import cv2 
import time 


#set params 

camera = cv2.VideoCapture(-1)
criteria = (
  cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
  30, 
  0.001
)
dimensions = tuple((7,7))
objp = np.zeros((dimensions[0]*dimensions[-1],3), np.float32)
objp[:,:2] = np.mgrid[0:dimensions[0],0:dimensions[-1]].T.reshape(-1,2)
objpoints = []
imgpoints = []

#objp, objpoints, imgpoints = init_points(dimensions)
while(True):
  print("reading a frame from attached camera")
  ret, frame = camera.read()
  print("frame dimensions: {}".format(frame.shape))
  print("convert frame to grayscale")
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  print("searching for chessboard")
  ret_, objcorners = cv2.findChessboardCorners(frame, dimensions, None)
  print("find function returns: {}".format(ret_))
  print("find function returns following object corners: \n{}".format(objcorners))

  if ret_ == True:
    print("located chessboard")
    objpoints.append(objp)
    imgcorners = cv2.cornerSubPix(
      frame,
      objcorners,
      (11,11),
      (-1,-1),
      criteria
    )
    imgpoints.append(objcorners)
    print("saved image points to master object")
    cv2.drawChessboardCorners(frame, dimensions, objcorners, ret_)
    time.sleep(5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeft = (0+50,0+50)
    fontScale = 0.75 
    fontColor = (255,255,255)
    lineType = 2 
    cv2.namedWindow('chessboard', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('chessboard', frame.shape[0],frame.shape[1])
    cv2.putText(frame, 'Found chessboard, try moving it', topLeft, font, fontScale, fontColor, lineType)
    cv2.imshow('chessboard', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    print("use view finder to center your chessboard")
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeft = (0+50,0+50)
    fontScale = 0.75
    fontColor = (255,255,255)
    lineType = 2 
    cv2.circle(frame, (frame.shape[0]/2,frame.shape[1]/2), int(200), (0,255,0), 2)
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0],frame.shape[1])
    cv2.putText(frame, 'Please center the chessboard in camera', topLeft, font, fontScale, fontColor, lineType)
    cv2.imshow('view_finder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

print("IMAGE_POINTS: \n{}".format(imgpoints))
print("OBJECT_POINTS: \n{}".format(objpoints))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame.shape[::-1],None, None)
print("Calibrated: {}".format(ret))
print("Matrix: {}".format(mtx))
print("Distortion: {}".format(dist))
print("Rotation Vector: {}".format(rvecs))
print("Translation Vector:: {}".format(tvecs))


camera.release()
cv2.destroyAllWindows()
      

    






