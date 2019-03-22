#!/usr/bin/env python
import cv2

cam = cv2.VideoCapture(-1)

supported_caps = [i for i in dir(cv2.cv) if i.startswith('CV_CAP')]

for cap in supported_caps:
  try:
    v = int(getattr(cv2.cv, cap))
    if v > 0:
      #value = cam.get(int(getattr(cv2.cv, cap)))
      print("{}:{}".format(cap, v))
    else:
      continue
  except cv2.error as e:
    print("skipped")



