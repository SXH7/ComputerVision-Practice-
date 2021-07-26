import cv2 as cv
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionConf = 0.5, trackingConf = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf


        self.mpPose =mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionConf, self.trackingConf)


    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lmList



def main():
    cap = cv.VideoCapture('videos/video3.mp4')
    ptime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        print(lmList)

        ctime = time.time()
        fps = 1/(ctime-ptime)

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        ptime = ctime

        cv.imshow('video', img)

        cv.waitKey(1)


if __name__ == "__main__":
    main()
