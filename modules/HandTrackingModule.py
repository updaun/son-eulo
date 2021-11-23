import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self,
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity, self.min_detection_confidence, self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw :
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findHandswithResult(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw :
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img, self.results

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                #if id == 4:
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0),cv2.FILLED) 

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax + 20, ymax + 20),
                (0,255,0), 2)

        return self.lmList, bbox

    def fingersUp(self, axis=False):
        fingers = []

        if axis == False:
            # Thumb
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[4]][1]:
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[4]][1]:
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers except Thumb
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        # axis = True( to detect LIKE gesture )
        else:
            # Thumb
            if self.lmList[self.tipIds[0]][2] < self.lmList[self.tipIds[0] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers except Thumb
            for id in range(1, 5):
                if self.lmList[self.tipIds[id]][1] > self.lmList[self.tipIds[id]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers

    def findLength(self, p1, p2):
        x1, y1 = self.lmList[p2][1:3]
        x2, y2 = self.lmList[p1][1:3]

        length = math.hypot(abs(x2-x1), abs(y2-y1))
        return length

    def findHandAngle(self, img, p1, p2, p3, draw=True):
        # 랜드마크 좌표 얻기
        # , x1, y1 = self.lmList[p1]
        x1, y1 = self.lmList[p1][1:3]
        x2, y2 = self.lmList[p2][1:3]
        x3, y3 = self.lmList[p3][1:3]

        # 각도 계산
        radian = math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2)
        angle = math.degrees(radian)

        if angle < 0:
            angle += 360

        #print(angle)
        # 점, 선 그리기
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(img, (x2,y2), (x3,y3), (255,255,255), 3)            
            cv2.circle(img, (x1,y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3,y3), 15, (0,0,255), 2)
            cv2.putText(img, str(int(angle)), (x2-50,y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2) 
            
        return angle

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,0,255), t)
            cv2.circle(img, (x1, y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0,0,255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break 

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()