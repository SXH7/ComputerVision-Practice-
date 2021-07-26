import cv2 as cv
import time
import PoseModule as pm

cap = cv.VideoCapture('videos/video3.mp4')
ptime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    print(lmList)

    ctime = time.time()
    fps = 1 / (ctime - ptime)

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    ptime = ctime

    cv.imshow('video', img)

    cv.waitKey(1)
