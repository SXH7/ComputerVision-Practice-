import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minimumDetectionConf=0.5):
        self.minimumDetectionConf = minimumDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, Draw=True):

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([bbox, detection.score])

                cv.rectangle(img, bbox, (255, 0, 255), 2)
                cv.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxes



def main():
    cap = cv.VideoCapture('videos/video1.mp4')
    ptime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        print(bboxes)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv.imshow('video', img)

        cv.waitKey(1)



if __name__ == "__main__":
    main()
