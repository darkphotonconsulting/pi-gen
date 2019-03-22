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



#cleanup
old_images = sorted(glob.glob('./tests/testCalibration/*.jpg'))
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
#dimensions = tuple((7,7))
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

while(True):
  #check threshold condition
  if count == threshold:
    print("i gathered {} chessboard frames, I will attempt to configure calibration data for the attached camera".format(threshold))
    break
  ret, frame = camera.read()

  #frame must be in grayscale colorspace for cv2.findChessboardCorners()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  print("please use the view finder to center your chessboard")

  #search..
  print("searching for chessboards in the last seen frame...")
  #various op flags can be zero or combination CALIB_CB_ADAPTIVE_THRESHOLD+CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FILTER_QUADS+CALIB_CB_FAST_CHECK
  #cv2.CALIB_CB_FAST_CHECK drastically speeds up the time required to find chessboard and thus drastically increases the view_finder FPS
  ret_, objcorners = cv2.findChessboardCorners(frame, dimensions, flags=cv2.CALIB_CB_FAST_CHECK)


  #chessboard
  if ret_ == True:
    #deal with threshold count
    count += 1 
    wkey = cv2.waitKey(1)
    print("located a chessboard in frame with object coordinates: \n{}".format(objcorners))
    #deal with points
    objpoints.append(objp)
    imgcorners = cv2.cornerSubPix(
      frame,
      objcorners,
      (11,11),
      (-1,-1),
      criteria
    )
    imgpoints.append(objcorners)

    print("calculated subpixels for grid within chessboard with dimensions {} with coordinates: \n{}".format(dimensions,imgcorners))
    cv2.imwrite("./tests/testCalibration/camera_precalib_frame%d.jpg" % count, frame)
    print("stored original frame @./tests/testCalibration/camera_precalib_frame%d.jpg" % count)
    cv2.drawChessboardCorners(frame, dimensions, objcorners, ret_)
    cv2.imwrite("./tests/testCalibration/camera_precalib_cornersdetected_frame%d.jpg" % count, frame)
    print("stored modified frame @./tests/testCalibration/camera_precalib_cornersdetected_frame%d.jpg" % count)


    #calculate centroid of image blob
    print("calculating frame centroid..")
    _, thresh = cv2.threshold(frame.copy(), 127, 255, 0)
    M = cv2.moments(frame) 
    cenX = int(M['m10'] / M['m00']) 
    cenY = int(M['m01'] / M['m00'])
    cencoord = (cenX, cenY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 

    #deal with view finder look & feel
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0],frame.shape[1])

    #deal with text/shape placement
    foundtxtX = (frame.shape[1] - foundtxtsize[0]) / 2
    foundtxtY = (frame.shape[0] - foundtxtsize[1]) / 2
    foundtxtcoord = (foundtxtX, foundtxtY) 
    txtw = foundtxtsize[0]
    txth = foundtxtsize[1] 
    
    cv2.putText(frame, foundtxt, (cenX-(txtw/2), cenY+(txth/2)), font, fontscale, txtcolor)

    #update display
    cv2.imshow('view_finder', frame)
    if wkey & 0xFF == ord('q'):
      sys.exit('received quit signal')
  else:
    wkey = cv2.waitKey(1)
    print("use view finder to center your chessboard")

    #deal with view finder look & feel
    #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
    cv2.namedWindow('view_finder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('view_finder', frame.shape[0],frame.shape[1])

    #deal with text/shape placement
    serchtxtX = (frame.shape[1] - serchtxtsize[0]) / 2
    serchtxtY = (frame.shape[0] - serchtxtsize[1]) / 2
    serchtxtcoord = (serchtxtX, serchtxtY) 
    txtw = serchtxtsize[0]
    txth = serchtxtsize[1] 

    #calculate centroid of image blob
    _, thresh = cv2.threshold(frame.copy(), 127, 255, 0)
    M = cv2.moments(frame) 
    cenX = int(M['m10'] / M['m00']) 
    cenY = int(M['m01'] / M['m00'])
    cencoord = (cenX, cenY)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
    print("frame centroid coords: {}x{}".format(cenX, cenY))
    cv2.circle(frame, cencoord, int(200), hex_to_bgr('FF4B4B'),  5)
    cv2.circle(frame, cencoord, int(150), hex_to_bgr('FF3333'),  5)
    cv2.circle(frame, cencoord, int(50),  hex_to_bgr('4C0000'), -1)
    cv2.putText(frame, serchtxt, (cenX-(txtw/2), cenY+(txth/2)), font, fontscale, txtcolor)

    #update display
    cv2.imshow('view_finder', frame)
    if wkey & 0xFF == ord('q'):
      sys.exit('received quit signal')

print("IMAGE_POINTS: \n{}".format(imgpoints))
print("OBJECT_POINTS: \n{}".format(objpoints))
print("FRAME SHAPE: \n{}".format(frame.shape[::-1]))
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (frame.shape[1], frame.shape[0]),None, None)
print("camera calibrated: {}".format(ret))
print("camera matrix: {}".format(mtx))
print("distortion coefficients: {}".format(dist))
print("rotation vector: {}".format(rvecs))
print("translation vector:: {}".format(tvecs))


camera.release()
cv2.destroyAllWindows()
#camera.open() 

#take another picture and 
      
#undistortion 
#ret, image = camera.read() 

#read in saved images 
test_images = sorted(glob.glob('./tests/testCalibration/camera_precalib_frame*.jpg'))

for test_image in test_images:
  print("calibration setup for {}".format(test_image))
  #general setup
  print("loading image..")
  img = cv2.imread(test_image)
  print("making a copy of image for remapping")
  img2 = img.copy()
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
  print("ROI VALUE: {}".format(roi))
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
  try:
    remapped = cv2.remap(img2, mapx, mapy, cv2.INTER_LINEAR)
  except cv2.error as e:
    print("not using image as remapping data is problematic, try more movement next time..")
    continue
  print("remapped image data with linear interpolation")
  #x,y,w,h = roi don't reset ROI
  print("cropped to ROI: x:{},y:{},w:{},h:{}".format(x, y, w, h))
  #remapped = remapped[y:y+h, x:x+w]
  cv2.imwrite(newpath, remapped)
  print("wrote {}".format(newpath))


#marshall camera matrix and distortion coefficients
np.savez("./CalibrationData/calib", mtx=mtx, newmtx=newmtx, dist=dist, rvecs=rvecs, tvecs=tvecs )
print("saved calibration data for camera and setting to disk, camera based operations will apply this calibration data based on user params, bye.")

print("calculating re-projection errors from object and image points")
mean_error = 0
for i in xrange(len(objpoints)):
  imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
  error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/ len(imgpoints2)
  mean_error += error

print "total errors calculated: ", mean_error/len(objpoints)
