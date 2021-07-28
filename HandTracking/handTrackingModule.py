import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import handTrackingModule as htm

wcam, hcam = 640, 480

cap = cv.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

ptime = 0

detector = htm.handDetector()



while True:

    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    print(lmList)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv.putText(img, str(int(fps)), (20, 50), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv.imshow('Video', img)
    cv.waitKey(10)
