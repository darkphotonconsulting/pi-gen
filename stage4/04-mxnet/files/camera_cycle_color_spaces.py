#!/usr/bin/env python
import numpy as np 
import cv2
import time 
#load cam
cap = cv2.VideoCapture(-1)

#process each frame in GRAY color space, display using local graphics toolkit, use `q` key to signal exit.
color = cv2.COLOR_BGR2RGB
font = cv2.FONT_HERSHEY_SIMPLEX 
txt = "BGR2RBG"
txtsize = cv2.getTextSize(txt, font, 1,2)[0] 
txtcolor = (255,255,255)
txtwidth = 5
while(True):
  ret, frame = cap.read()
  print('bgr (default)')
  frame = cv2.cvtColor(frame, color)
  #determine text coordinates with respect to frame dimensions & text size.
  txtX = (frame.shape[1] - txtsize[0]) / 2
  txtY = (frame.shape[0] - txtsize[1]) / 2
  txtcoord = (txtX, txtY)
  cv2.putText(frame, txt, txtcoord, font, 1, txtcolor)

  cv2.imshow('view finder', frame)
  key = cv2.waitKey(1) 
  if key & 0xFF == ord('q'):
    print('exit')
    break
  elif key & 0xFF == ord('g'): 
    print('gray')
    color = cv2.COLOR_BGR2GRAY
    txt = "BGR2GRAY"
  elif key & 0xFF == ord('h'): 
    print('hsv')
    color = cv2.COLOR_BGR2HSV
    txt = "BGR2HSV"
  elif key & 0xFF == ord('r'): 
    print('rgb')
    color = cv2.COLOR_BGR2RGB
    txt = "BGR2RGB"
  elif key & 0xFF == ord('l'): 
    print('lab')
    color = cv2.COLOR_BGR2LAB
    txt = "BGR2LAB"
  elif key & 0xFF == ord('y'): 
    print('yuv')
    color = cv2.COLOR_BGR2YUV
    txt = "BGR2YUV"
  elif key & 0xFF == ord('u'): 
    print('luv')
    color = cv2.COLOR_BGR2LUV
    txt = "BGR2LUV"
  else:
    print('blah')
    


#cleanup
cap.release()
cv2.destroyAllWindows()
