#Remove after first execution
import setup

import random
import time

import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector

# Open Webcam
cap = cv2.VideoCapture(0)
# Set with of capture
cap.set(3, 640)
# Set height of capture
cap.set(4, 480)

# Hand Detect
detector = HandDetector(maxHands=1)

# Timer
timer = 0

# Flags
stateResult = False
startGame = False

# Score
scores = [0, 0]  # [AI, Player]

while True:
    # Get Background image
    imgBG = cv2.imread("res/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    # Find Hands
    hands, img = detector.findHands(imgScaled)

    if startGame and (not stateResult):
        #if not stateResult:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    hand = hands[0]
                    playerMove = None
                    fingers = detector.fingersUp(hand)

                    # Rock
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    # Paper
                    if fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    # Scissors
                    if fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    randomNumer = random.randint(1, 3)
                    imgAI = cv2.imread(f'res/{randomNumer}.png', cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                    # Player wins
                    if playerMove != randomNumer:
                        if (playerMove == 1 and randomNumer == 3) or \
                                (playerMove == 2 and randomNumer == 1) or (playerMove == 3 and randomNumer == 2):
                            scores[1] += 1
                        else:
                            scores[0] += 1

                    if playerMove:
                        print(playerMove)
                    else:
                        print('Unrecognised')

    # Place webcam image on BG
    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        imgBG = imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    # Change score
    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 6)

    # Show image
    cv2.imshow("BG", imgBG)

    key = cv2.waitKey(1)

    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False