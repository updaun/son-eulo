import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import modules.HolisticModule as hm
import math
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Holistic 객체 생성
detector = hm.HolisticDetector()

text = ''

while True:

    # defalut BGR img
    success, img = cap.read()
    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHolistic(img, draw=True)
    
    # left_hand_lmList = detector.findLefthandLandmark(img, draw=False)
    right_hand_lmList = detector.findRighthandLandmark(img, draw=True)

    # 감지가 되었는지 확인하는 구문
    if len(right_hand_lmList) != 0:

        x1, y1 = right_hand_lmList[12][1:3]
        x2, y2 = right_hand_lmList[9][1:3]

        x3, y3 = right_hand_lmList[16][1:3]
        x4, y4 = right_hand_lmList[13][1:3]

        # 각도 계산
        radian_1 = math.atan2(y2-y1, x2-x1)
        angle_1 = int(math.degrees(radian_1))

        radian_2 = math.atan2(y4-y3, x4-x3)
        angle_2 = int(math.degrees(radian_2))

        if angle_1 < 0:
            angle_1 += 360

        if angle_2 < 0:
            angle_2 += 360

        result = angle_2 - angle_1
        if result < 0:
            text = 'ㅌ'
        elif 0 < result < 20:
            text = 'ㄹ'


    # Get status box
    cv2.rectangle(img, (0,0), (100, 60), (245, 117, 16), -1)

    # Display Probability
    cv2.putText(img, 'STATUS'
                , (15,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 한글 적용
    b,g,r,a = 255,255,255,0
    # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
    fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(fontpath, 35)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 15), f'{text}', font=font, fill=(b,g,r,a))
    img = np.array(img_pil)
    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()
    