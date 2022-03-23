from __future__ import print_function
from re import I
import numpy as np
import cv2 as cv
import argparse
from pynput.keyboard import Key, Controller

centerWidth = 75
# 320 is center
leftThreshold = int(320 + (centerWidth/2))
rightThreshold = int(320 - (centerWidth/2))

toggleKeyPress = False

cap = cv.VideoCapture(0)
keyboard = Controller()

if not cap.isOpened():
  print('Cannot open camera')
  exit()

def checkForLean(headCenter):
  if headCenter[0] < rightThreshold:
    # Lean right
    return 2
  elif headCenter[0] > leftThreshold:
    # Lean left
    return 0
  # No lean
  return 1

def handleLeanInput(leanDirection):
  if not toggleKeyPress:
    return
  if leanDirection == 0:
    # Lean left
    keyboard.release(']')
    keyboard.press('[')
  if leanDirection == 1:
    # Stop leaning
    keyboard.release(']')
    keyboard.release('[')
  if leanDirection == 2:
    # Lean right
    keyboard.release('[')
    keyboard.press(']')

# Main frame analysis function
def detectLean(frame):
  # Split feed into a grayscale copy for ML
  frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  frame_gray = cv.equalizeHist(frame_gray)

  # Detect faces and determine lean
  lean = 'undef'
  faces = face_cascade.detectMultiScale(frame_gray)
  i = 0
  for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    lean = 1
    # Only look at the first face and make sure it is wider than 100px (filters false positives)
    if i == 0 and w > 100:
      # Check center location of face
      lean = checkForLean(center)
      # Handle keyboard input
      handleLeanInput(lean)
      # Draw circle around face for debugging
      if lean == 1:
        cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (0, 255, 0), 4)
      elif lean == 0:
        cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
      elif lean == 2:
        cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (0, 0, 255), 4)
    i+=1

  # Draw center lines
  cv.line(frame,(rightThreshold,0),(rightThreshold,500),(255,0,0),1)
  cv.line(frame,(leftThreshold,0),(leftThreshold,500),(255,0,0),1)
  # Write text showing on/off state
  font = cv.FONT_HERSHEY_SIMPLEX
  if toggleKeyPress:
    cv.putText(frame,'ON (press "=" to turn off key inputs)',(10,450), font, 1,(0,255,0),2,cv.LINE_AA)
  else:
    cv.putText(frame,'OFF (press "=" to turn on key inputs)',(10,450), font, 1,(255,0,0),2,cv.LINE_AA)

  # display the resulting frame
  cv.imshow('frame', frame)

# ML Data setup stuff
parser = argparse.ArgumentParser(description='Webcam lean detector')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='./haarcascade_frontalface_alt.xml')
# parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='./haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()
face_cascade_name = args.face_cascade
# eyes_cascade_name = args.eyes_cascade
face_cascade = cv.CascadeClassifier()
# eyes_cascade = cv.CascadeClassifier()

#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
camera_device = args.camera

# Main loop
while True:
  # Capture frame by frame
  ret, frame = cap.read()
  # if frame is read correctly ret is true
  if not ret:
    print('Cant recieve frame (stream end?). Exiting ...')
    break
  # Logic
  detectLean(frame)
  # Handle key presses
  keyDown = cv.waitKey(1)
  if keyDown == ord('='):
    toggleKeyPress = not toggleKeyPress
  if keyDown == ord('q'):
    break

# when everything is done release the capture
cap.release()
cv.destroyAllWindows()
