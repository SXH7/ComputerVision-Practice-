import cv2 as cv
import mediapipe as mp
import time

mpPose =mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

cap = cv.VideoCapture('videos/video3.mp4')
ptime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv.circle(img, (cx, cy), 5, (0, 0, 255), cv.FILLED)

    ctime = time.time()
    fps = 1/(ctime-ptime)

    cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    ptime = ctime

    cv.imshow('video', img)

    cv.waitKey(1)
