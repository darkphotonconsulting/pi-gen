import os, sys, glob
import numpy as np 
import cv2 
import time 

#setup required vars
camera = cv2.VideoCapture(-1)
dimensions = tuple((7,7))
criteria = (
  cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
  30,
  0.001
)
#setup required point data
objp = np.zeros((dimensions[0]*dimensions[-1],3), np.float32)
objp[:,:2] = np.mgrid[0:dimensions[0],0:dimensions[-1]].T.reshape(-1,2)
objpoints = [] 
imgpoints = []

#deal with finding chess board
while(True):
  print("reading a frame from web camera")
  ret, frame = camera.read()
  print("frame dimensions: {}".format(frame.shape))
  print("convert frame to grayscale")
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  print("showing view port")
  #cv2.imshow('view_port', frame)
  #if cv2.waitKey(1) & 0xFF == ord('q'): 
  #  break
  print("try to find chessboard")
  ret_, objcorners = cv2.findChessboardCorners(frame, dimensions, None)
  print("find function returns: {}".format(ret_))
  print("find function returns following object corners: {}".format(objcorners))
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
    # imgpoints.append(imgcorners)
    imgpoints.append(objcorners)
    print("drawing over chessboard")
    cv2.drawChessboardCorners(frame, dimensions, objcorners, ret_)
    time.sleep(5)
    cv2.imshow('view_finger', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break
    
  else:
    print("showing view port")
    cv2.imshow('view_finder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
      break

    
