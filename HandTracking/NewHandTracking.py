import cv2 as cv
#import mediapipe as mp
import time
from HandTracking import handTrackingModule as htm

ptime = 0
ctime = 0
cap = cv.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[4])

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 225, 0), 3)

    cv.imshow('image', img)
    cv.waitKey(1)