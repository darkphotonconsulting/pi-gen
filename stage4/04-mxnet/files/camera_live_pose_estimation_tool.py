import os, sys, glob, argparse
import numpy as np 
import cv2 
import time 

parser = argparse.ArgumentParser(description='ML Pi Camera Calibration Tool')
parser.add_argument('--dimensions', help='grid dimensions (default: 6x7)', action='store', nargs='*',required=False, default=[7,6] )
parser.add_argument('--threshold', help='# of images to capture for calibration (default: 20)', action='store',required=False, default=20 )

args = parser.parse_args()

#hex2rgb 
def hex_to_bgr(h):
  r = tuple(int(h[i:i+2], 16) for i in (0,2,4))  
  return r[::-1]



dimensions = tuple(args.dimensions)
objp = np.zeros((dimensions[0]*dimensions[-1],3), np.float32)
objp[:,:2] = np.mgrid[0:dimensions[0],0:dimensions[-1]].T.reshape(-1,2)
objpoints = []
imgpoints = []
count = 0 
threshold = int(args.threshold)

font = cv2.FONT_HERSHEY_TRIPLEX
fontscale = 0.88
thickness = 3 
baseline = 0 
serchtxt = "center board in view finder"
serchtxtsize = cv2.getTextSize(serchtxt, font, 1, 2)[0]
foundtxt = "jiggle board around in view finder"
foundtxtsize = cv2.getTextSize(foundtxt, font, 1, 2)[0]
txtcolor = (255,255,255)

criteria = (
  cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
  30, 
  0.001
)
#convenience vars
cols_and_rows = (dimensions[0], dimensions[1])
rows_and_cols = (dimensions[1],dimensions[0])

objp = np.zeros((dimensions[0]*dimensions[-1],3), np.float32)
objp[:,:2] = np.mgrid[0:dimensions[0],0:dimensions[-1]].T.reshape(-1,2)
objpoints = []
imgpoints = []
count = 0 
threshold = int(args.threshold)

camera = cv2.VideoCapture(-1) 
with np.load('CalibrationData/calib.npz') as X:
  mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

while(True):
  ret, frame = camera.read() 
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  ret_, objcorners = cv2.findChessboardCorners(frame, dimensions, flags=cv2.CALIB_CB_FAST_CHECK)

  if ret_ == True:
    wkey = cv2.waitKey(1)
    objpoints.append(objp)
    imgcorners = cv2.cornerSubPix(
      frame,
      objcorners,
      (11,11),
      (-1,-1),
      criteria
    )
    imgpoints.append(objcorners)
    cv2.drawChessboardCorners(frame, dimensions, objcorners, ret_)
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #switch back to color frame
    #calculate frame centroid 
    _, thresh = cv2.threshold(frame, 127, 255, 0)
    M = cv2.moments(frame)
    cenX = int(M['m10'] / M['m00'])
    cenY = int(M['m01'] / M['m00'])
    cencoord = (cenX, cenY)
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0], frame.shape[1])
    txtw = foundtxtsize[0] 
    txth = foundtxtsize[1] 
    cv2.putText(frame, foundtxt, (cenX-(txtw/2), cenY-(txth/2)), font, fontscale, txtcolor)
    print('projecting axis on rotated plane')
    axis = np.float32([ [3,0,0], [0,3,0], [0,0,-3] ]).reshape(-1,3)
    rvecs, tvecs, inliners = cv2.solvePnPRansac(objp, objcorners, mtx, dist)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    corner = tuple(objcorners[0].ravel())
    imgpoint1 = tuple(imgpts[0].ravel())
    imgpoint2 = tuple(imgpts[1].ravel())
    imgpoint3 = tuple(imgpts[2].ravel())
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) #switch back to color frame
    cv2.line(frame, corner, imgpoint1, (255,0,0), 5)
    cv2.line(frame, corner, imgpoint2, (0,255,0), 5)
    cv2.line(frame, corner, imgpoint3, (0,0,255), 5)
    cv2.imshow('view_finder', frame)
    if wkey & 0xFF == ord('q'): 
      sys.exit('received quit signal')
  else:
    wkey = cv2.waitKey(1) 
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0], frame.shape[1])
    txtw = serchtxtsize[0]
    txth = serchtxtsize[1]
    #calculate frame centroid 
    _, thresh = cv2.threshold(frame, 127, 255, 0)
    M = cv2.moments(frame)
    cenX = int(M['m10'] / M['m00'])
    cenY = int(M['m01'] / M['m00'])
    cencoord = (cenX, cenY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.circle(frame, cencoord, int(200), hex_to_bgr('FF4B4B'), 5)
    cv2.circle(frame, cencoord, int(150), hex_to_bgr('FF3333'), 5)
    cv2.circle(frame, cencoord, int(50),  hex_to_bgr('4C0000'), 5)
    cv2.putText(frame, serchtxt, (cenX-(txtw/2), cenY+(txth/2)), font, fontscale, txtcolor)

    cv2.imshow('view_finder', frame)
    if wkey & 0xFF == ord('q'):
      sys.exit('received quit signal')






