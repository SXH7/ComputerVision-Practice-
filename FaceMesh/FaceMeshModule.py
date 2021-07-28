import cv2 as cv
import mediapipe as mp
import time

class FaceMeshDetector():

    def __init__(self, staticMode = False, maxFaces = 2, minDetectionConf = 0.5, minTrackingConf = 0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionConf
        self.minTrackCon = minTrackingConf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        #self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS)
                face = []
                for lm in faceLms.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y * ih)
                    #print(x, y)
                    face.append([x, y])
            faces.append(face)
            return img, faces




def main():
    cap = cv.VideoCapture('videos/video3.mp4')
    ptime = 0

    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(len(faces))

        ctime = time.time()
        fps = 1/(ctime-ptime)
        cv.putText(img, str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        ptime = ctime
        cv.imshow("video", img)
        cv.waitKey(1)




if __name__ == "__main__":
    main()
