#!/usr/bin/env python
import numpy as np 
import cv2

#load cam
cap = cv2.VideoCapture(-1)

#process each frame in GRAY color space, display using local graphics toolkit, use `q` key to signal exit.
while(True):
  ret, frame = cap.read()
  #color = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

#cleanup
cap.release()
cv2.destroyAllWindows()
