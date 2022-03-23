from __future__ import print_function
from re import I
import numpy as np
import cv2 as cv
import argparse
from pynput.keyboard import Key, Controller

deltaTolerance = 5
eyeHeights = []

cap = cv.VideoCapture(0)
keyboard = Controller()

if not cap.isOpened():
  print('Cannot open camera')
  exit()

def checkEyeHeight(leftEye, rightEye):
  delta = leftEye - rightEye
  if delta > deltaTolerance:
    # Lean right
    # keyboard.release('[')
    # keyboard.press(']')
    return 2
  elif delta < (deltaTolerance*-1):
    # Lean left
    # keyboard.release(']')
    # keyboard.press('[')
    return 0
  # keyboard.release(']')
  # keyboard.release('[')
  return 1
      
      

# Main frame manipulation function
def detectLean(frame):
  # our operation on the frame come here
  frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  frame_gray = cv.equalizeHist(frame_gray)

  #-- Detect faces
  lean = 'undef'
  faces = face_cascade.detectMultiScale(frame_gray)
  for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = frame_gray[y:y+h,x:x+w]
    #-- In each face, detect eyes
    eyes = eyes_cascade.detectMultiScale(faceROI)
    i = 0
    eyeHeights.clear()
    for (x2,y2,w2,h2) in eyes:
      eye_center = (x + x2 + w2//2, y + y2 + h2//2)
      if i==0 or i==1:
          eyeHeights.append(eye_center)
      radius = int(round((w2 + h2)*0.25))
      frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
      i+=1
    
    if len(eyeHeights) == 2:
      lean = checkEyeHeight(eyeHeights[0][1], eyeHeights[1][1])
  
  cv.line(frame,(320,0),(320,500),(255,0,0),2)
  font = cv.FONT_HERSHEY_SIMPLEX
  cv.putText(frame,str(lean),(10,450), font, 1,(255,0,0),2,cv.LINE_AA)

  # display the resulting frame
  cv.imshow('frame', frame)

# ML Data setup stuff
parser = argparse.ArgumentParser(description='Webcam lean detector')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_alt.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='./haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

while True:
  # Capture frame by frame
  ret, frame = cap.read()
  # if frame is read correctly ret is true
  if not ret:
    print('Cant recieve frame (stream end?). Exiting ...')
    break
  detectLean(frame)
  if cv.waitKey(1) == ord('q'):
    break

# when everything is done release the capture
cap.release()
cv.destroyAllWindows()
