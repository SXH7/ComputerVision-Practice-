import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture('videos/video3.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)

ptime = 0
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS)
            for lm in faceLms.landmark:
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y * ih)
                print(x, y)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    ptime = ctime
    cv.imshow("video", img)
    cv.waitKey(1)
