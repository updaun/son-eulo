import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import modules.HandTrackingModule as htm


# video input 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Holistic 객체 생성
detector = htm.handDetector(max_num_hands=1)

cnt10 = 0
text_cnt = 0
dcnt = 0
min_detec = 10
max_detec = 30
num_lst = [11, 15, 16]

flag = False

while True:
    number = ''

    # defalut BGR img
    success, img = cap.read()
    # mediapipe를 거친 이미지 생성 -> img
    img = detector.findHands(img, draw=True)
    
    hand_lmList, _ = detector.findPosition(img, draw=True)

    # 감지가 되었는지 확인하는 구문
    if len(hand_lmList) != 0:

        # x축을 기준으로 손가락 리스트
        right_hand_fingersUp_list_a0 = detector.fingersUp(axis=False)
        # y축을 기준으로 손가락 리스트
        right_hand_fingersUp_list_a1 = detector.fingersUp(axis=True)
        # 엄지 끝과 검지 끝의 거리 측정
        thumb_index_length = detector.findLength(4, 8)

        # 검지 구부림 감지를 위한 각도 변수
        index_finger_angle_1 = int(detector.findHandAngle(img, 8, 9, 5, draw=False))
        index_finger_angle_2 = int(detector.findHandAngle(img, 8, 13, 5, draw=False))
        index_finger_angle_3 = int(detector.findHandAngle(img, 8, 17, 5, draw=False))
        total_index_angle = index_finger_angle_1 + index_finger_angle_2 + index_finger_angle_3
        
        # 중지 구부림 감지를 위한 각도 변수
        middle_finger_angle_1 = 360 - int(detector.findHandAngle(img, 12, 5, 9, draw=False))
        middle_finger_angle_2 = int(detector.findHandAngle(img, 12, 13, 9, draw=False))
        middle_finger_angle_3 = int(detector.findHandAngle(img, 12, 17, 9, draw=False))
        total_middle_angle = middle_finger_angle_1 + middle_finger_angle_2 + middle_finger_angle_3
        
        # 손바닥이 보임, 수향이 위쪽
        if hand_lmList[5][1] > hand_lmList[17][1] and hand_lmList[4][2] > hand_lmList[8][2]:
            if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmList[8][2] < hand_lmList[7][2]:
                number = 1
            elif right_hand_fingersUp_list_a0 == [0, 1, 1, 0, 0]:
                number = 2
            elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 0] or right_hand_fingersUp_list_a0 == [1, 1, 1, 1, 0]:
                number = 3
            elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 1]:
                number = 4
            elif right_hand_fingersUp_list_a0 == [1, 0, 1, 1, 1] and thumb_index_length < 30:
                number = 10  # 동그라미 10

        # 손바닥이 보임
        if hand_lmList[5][1] > hand_lmList[17][1]:
            if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                if right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
                    number = 0
                else:
                    number = 5
            # 손가락을 살짝 구부려 10과 20 구분
            if right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[2:] == [0, 0, 0] and total_index_angle < 140 and total_middle_angle > 300:
                number = 10
                cnt10 += 1
            elif right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[3:] == [0, 0] and total_index_angle < 140 and total_middle_angle < 150:
                number = 20

        # 손등이 보임, 수향이 몸 안쪽으로 향함, 엄지가 들려 있음
        if hand_lmList[5][2] < hand_lmList[17][2] and hand_lmList[4][2] < hand_lmList[8][2]:
            if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
                number = 6
            elif right_hand_fingersUp_list_a1 == [1, 1, 1, 0, 0]:
                number = 7
            elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 0]:
                number = 8
            elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
                number = 9

        # 손등이 보이고, 수향이 몸 안쪽으로 향함
        if hand_lmList[5][2] < hand_lmList[17][2] and hand_lmList[1][2] < hand_lmList[13][2]:
            # 엄지가 숨어짐
            if hand_lmList[4][2] + 30 > hand_lmList[8][2]:
                if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 12
                elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 13
                elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 14
            # 엄지가 보임
            else:
                if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 17
                elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 18
                elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmList[8][1] <= hand_lmList[6][1] + 20:
                    number = 19    

        # 10이 계속 인식된 경우 10으로 표현
        if cnt10 > (max_detec - min_detec):
            number = 10
            flag = True
            # print("clear")
            # dcnt = 0
            
        # 10이 인식되었다가 1, 5, 6이 인식되는 경우    
        elif cnt10 > min_detec:
            if hand_lmList[5][1] > hand_lmList[17][1] and hand_lmList[4][2] > hand_lmList[8][2]:
                if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmList[8][2] < hand_lmList[7][2]:
                    dcnt += 1
                    number = ''
                    if max_detec > dcnt > min_detec:
                        number = 11
                    elif dcnt > max_detec+10:
                        number = ''
                        cnt10 = 0
                        dcnt = 0
                        # print("clear")
            elif hand_lmList[5][1] > hand_lmList[17][1]:
                if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                    dcnt += 1
                    number = ''
                    if max_detec > dcnt > min_detec:
                        number = 15
                    elif dcnt > max_detec+10:
                        number = ''
                        cnt10 = 0
                        dcnt = 0
            elif hand_lmList[5][2] < hand_lmList[17][2] and hand_lmList[4][2] < hand_lmList[8][2]:
                if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
                    dcnt += 1
                    number = ''
                    if max_detec > dcnt > min_detec:
                        number = 16
                    elif dcnt > max_detec+10:
                        number = ''
                        cnt10 = 0
                        dcnt = 0
                        
        if number in num_lst:
            flag = True

    img = cv2.flip(img, 1)

    # Get status box
    cv2.rectangle(img, (0,0), (100, 60), (245, 117, 16), -1)

    # Display Probability 
    cv2.putText(img, 'NUMBER'
                , (2,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(number)
                , (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if flag:
        text_cnt += 1
        if text_cnt % max_detec == 0:
            cnt10 = 0
            text_cnt = 0
            dcnt = 0
            flag = False
    
    # 손이 감지 안되었을 때
    if len(hand_lmList) == 0:
        cnt10 = 0
        text_cnt = 0
        dcnt = 0
        flag = False
                    
    # img를 우리에게 보여주는 부분
    cv2.imshow("Image", img)

        
    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

cap.release()
cv2.destroyAllWindows()