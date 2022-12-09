
#Remove after first execution
import setup

#Loading dependencies

import cv2
import mediapipe as mp
import time

#Full body pose estimation Example
"""
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('videoplayback.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pTime = 0

while True:
  success, img = cap.read()
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = pose.process(imgRGB)
  #print(results.pose_landmarks)
  if results.pose_landmarks:
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
      h, w,c = img.shape
      print(id, lm)
      cx, cy = int(lm.x*w), int(lm.y*h)
      cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)

      cTime = time.time()
      fps = 1/((cTime)-pTime)
      pTime = cTime

      cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
      cv2.imshow("Image", img)
      cv2.waitKey(1)
"""

###Hand pose estimation###

#Change this to use video or webcam
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videoplayback_h.mp4')

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                if id ==0:
                  cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
