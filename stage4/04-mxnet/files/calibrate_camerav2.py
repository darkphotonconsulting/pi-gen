import os, sys, glob, argparse
import numpy as np 
import cv2 
import time 


#cleanup
old_images = glob.glob('./tests/testCalibration/*.jpg')
for image in old_images:
  os.remove(image)
  print("removed {}".format(image))

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
count = 0 
threshold = 20

#objp, objpoints, imgpoints = init_points(dimensions)
while(True):
  if count == threshold:
    print("gathered {} Chessboard frames, configuring calibration data..".format(threshold))
    break
  print("reading frame from video capture object")
  ret, frame = camera.read()
  print("current frame dimensions: {}".format(frame.shape))
  width = camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
  height = camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
  print("frame width: {} frame height: {}".format(width, height))
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  print("converted frame to grayscale")
  #print("please use the view finder to center your chessboard")
  #font = cv2.FONT_HERSHEY_SIMPLEX
  #topLeft = (0+50,0+50)
  #fontScale = 0.75
  #fontColor = (255,255,255)
  #lineType = 2 
  #cv2.circle(frame, (frame.shape[0]/2,frame.shape[1]/2), int(200), (0,255,0), 2)
  #cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
  #cv2.resizeWindow('view_finder', frame.shape[0],frame.shape[1])
  #cv2.putText(frame, 'Please center the chessboard in camera', topLeft, font, fontScale, fontColor, lineType)
  #cv2.imshow('view_finder', frame)
  #if cv2.waitKey(1) & 0xFF == ord('q'):
  #  break
  print("searching for chessboard")
  ret_, objcorners = cv2.findChessboardCorners(frame, dimensions, None)
  print("find function returns: {}".format(ret_))
  print("find function returns following object corners: \n{}".format(objcorners))

  if ret_ == True:
    time.sleep(0.5)
    print("located chessboard")
    objpoints.append(objp)
    print("saved object points to master object")
    imgcorners = cv2.cornerSubPix(
      frame,
      objcorners,
      (11,11),
      (-1,-1),
      criteria
    )
    imgpoints.append(objcorners)
    #cleanup any view finder windows. 
    if cv2.getWindowProperty('view_finder', 1) > 0: 
      print("cleaning up view_finder")
      cv2.destroyWindow('view_finder')
    print("saved chessboard corners to master object")
    cv2.imwrite("./tests/testCalibration/camera_precalib_frame%d.jpg" % count, frame)
    print("saved pristene image to ./tests/testCalibration/camera_precalib_frame%d.jpg" % count)
    cv2.drawChessboardCorners(frame, dimensions, objcorners, ret_)
    cv2.imwrite("./tests/testCalibration/camera_precalib_cornersdetected_frame%d.jpg" % count, frame)
    print("saved pristene image to ./tests/testCalibration/camera_precalib_cornersdetected_frame%d.jpg" % count)
    #time.sleep(5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeft = (0+50,0+50)
    fontScale = 0.75 
    fontColor = (255,255,255)
    lineType = 2 
    cv2.namedWindow('view_chessboard', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_chessboard', frame.shape[0],frame.shape[1])
    cv2.putText(frame, 'Found chessboard, try moving it', topLeft, font, fontScale, fontColor, lineType)
    cv2.imshow('view_chessboard', frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    time.sleep(0.5)
    print("use view finder to center your chessboard")
    if cv2.getWindowProperty('view_chessboard', 1) > 0: 
      print("cleaning up view_chessboard")
      cv2.destroyWindow('view_chessboard')
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeft = (0+50,0+50)
    fontScale = 0.75
    fontColor = (255,255,255)
    lineType = 2 
    cv2.circle(frame, (frame.shape[0]/2,frame.shape[1]/2), int(200), (50,255,50), 5)
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0],frame.shape[1])
    cv2.putText(frame, 'Please center the chessboard in camera', topLeft, font, fontScale, fontColor, lineType)
    cv2.imshow('view_finder', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

#print("IMAGE_POINTS: \n{}".format(imgpoints))
#print("OBJECT_POINTS: \n{}".format(objpoints))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame.shape[::-1],None, None)
print("Camera Calibrated: {}".format(ret))
print("Camera Matrix: {}".format(mtx))
print("Distortion Coefficients: {}".format(dist))
print("Rotation Vector: {}".format(rvecs))
print("Translation Vector:: {}".format(tvecs))


camera.release()
cv2.destroyAllWindows()
#camera.open() 

#take another picture and 
      
#undistortion 
#ret, image = camera.read() 

#read in saved images 
test_images = glob.glob('./tests/testCalibration/camera_precalib_frame*.jpg')

for test_image in test_images:
  print("calibration setup for {}".format(test_image))
  #general setup
  print("loading image..")
  img = cv2.imread(test_image)
  img_file_name = os.path.basename(test_image)
  img_file_name = img_file_name.replace('precalib','calibrated')
  img_file_name = "cropped_" + img_file_name
  h, w = img.shape[:2]
  print("image height: {} image width: {}".format(h,w))

  #undistort
  print("calculate optimal camera matrix")
  newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
  print("optimal camera matrix calculated: \n{}\n ROI calculated for copping: \n{}\n... will perform shortest-path undistort ...\n".format(newmtx,roi))
  dst = cv2.undistort(img, mtx, dist, None, newmtx)
  x,y,w,h = roi 
  dst = dst[y:y+h, x:x+w]
  print("cropped to ROI: \nx:{},y:{},w:{},h:{}".format(x, y, w, h))
  newpath = './tests/testCalibration/' + 'undistorted_' +  img_file_name 
  cv2.imwrite(newpath, dst)
  print("wrote {}".format(newpath))


  #re-map original
  newpath = './tests/testCalibration/' + 'remapped_' +  img_file_name 
  mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newmtx, (w,h), 5)
  print("distortion rectification maps calculated\nx:\n{}\ny:\n{}".format(mapx,mapy))
  dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
  print("remapped image data with linear interpolation")
  x,y,w,h = roi 
  print("cropped to ROI: x:{},y:{},w:{},h:{}".format(x, y, w, h))
  dst = dst[y:y+h, x:x+w]
  cv2.imwrite(newpath, dst)
  print("wrote {}".format(newpath))


#marshall camera matrix and distortion coefficients
np.savez("./CalibrationData/calib", mtx=mtx, newmtx=newmtx, dist=dist, rvecs=rvecs, tvecs=tvecs )
print("saved calibration data for camera and setting to disk, camera based operations will apply this calibration data based on user params, bye.")






