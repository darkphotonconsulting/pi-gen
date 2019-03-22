#!/usr/bin/env python
import numpy as np 
import cv2
import os, sys
import argparse

parser = argparse.ArgumentParser(description='ML Pi Camera Test Tool')
parser.add_argument('--list_color_spaces', help='List Usable Color Spaces for Color Conversion', action='store_true', required=False, default=False)
parser.add_argument('--color_space', help='Select Color Space to Convert Output to, if none selected, output is raw', action='store', required=False, default="")
args = parser.parse_args() 



def list_color_spaces():
  colspaces = [i for i in dir(cv2) if i.startswith('COLOR_')]
  print colspaces
  return colspaces

def capture(colspace):
  cap = cv2.VideoCapture(-1)
  while(True):
    ret, frame = cap.read()
    if colspace == "":
      cv2.imshow('raw', frame)
    else:
      frame = cv2.cvtColor(frame, getattr(cv2, colspace))
      cv2.imshow(colspace, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

def main(a): 
  if a.list_color_spaces:
    cspace = list_color_spaces()
    print("\n".join(cspace))
    sys.exit()
  
  capture(a.color_space)
  


main(args)
