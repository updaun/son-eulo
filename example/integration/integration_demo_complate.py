import cv2
import copy
import csv
import itertools
from collections import Counter, deque

import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from PIL import ImageFont, ImageDraw, Image

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
# from models import KeyPointClassifier
from models import PointHistoryClassifier
import modules.HandTrackingModule as htm

from hangul_utils import split_syllable_char, split_syllables, join_jamos

# Hand 객체 생성
detector = htm.handDetector(max_num_hands=1)

# ------------------- 모델 ------------------- #

## m1 방향에 따라 분류(손바닥 위)
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅇ','ㅎ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
interpreter_m1 = tf.lite.Interpreter(model_path="models/modelvector/model1.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
actions_m2 = ['ㅇ','ㅎ','ㅗ','ㅚ','ㅛ']
interpreter_m2 = tf.lite.Interpreter(model_path="models/modelvector/model2.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ']
interpreter_m3 = tf.lite.Interpreter(model_path="models/modelvector/model3.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
interpreter_m4 = tf.lite.Interpreter(model_path="models/modelvector/model4.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅡ','ㅢ']
interpreter_m5 = tf.lite.Interpreter(model_path="models/modelvector/model5.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()

# -------------------------------------------------- #

# 덮어씌우는 이미지 리스트
folderPath = "images/button_image"
myList = os.listdir(folderPath)

# 덮어씌우는 이미지 리스트
overlayList =[]

# Header 폴더에 image를 상대경로로 지정
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)



def main(mode, mode_count, button_overlay, delete_count, delete_button_overlay, s2_lst_remove):
    # action Variable
    cnt10 = 0
    text_cnt = 0
    dcnt = 0
    min_detec = 10
    max_detec = 30
    num_lst = [11, 15, 16]
    flag = False

    # Korean Variable
    seq_length = 10
    seq = []
    action_seq = []
    last_action = None
    this_action = ''
    select_model = ''
    wrist_angle = 0
    confidence = 0.9
    action = ''
    
    # User Interface Variable
    button_overlay = overlayList[1]

    # Keyboard Variable
    cnt = 0
    jamo_li = deque()
    jamo_join_li = deque()

    status_cnt_conf = 10
    status_lst = deque(['Stop']*status_cnt_conf, maxlen=status_cnt_conf)

    M = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅣ', 'ㅗ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅟ', 'ㅠ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅡ', 'ㅢ', 'ㅘ', 'ㅙ', 'ㅝ', 'ㅞ']
    J = ["ㄱ", "ㅅ", "ㅈ", "ㅊ", "ㅋ", "ㄴ", "ㄷ", "ㄹ", "ㅌ", "ㅁ", "ㅂ", "ㅍ", "ㅇ", "ㅎ", "ㄲ", "ㅆ", "ㅉ", "ㄸ", "ㅃ"]
    JJ_dict = {
        "ㄱ":"ㄲ",
        "ㅅ":"ㅆ",
        "ㅈ":"ㅉ",
        "ㄷ":"ㄸ",
        "ㅂ":"ㅃ"
        }
    siot = ['ㅅ', 'ㅆ']
    MM_lst = ['ㅗ', 'ㅜ']
    MM_dict = {
        "ㅏ":"ㅘ",
        "ㅐ":"ㅙ",
        "ㅓ":"ㅝ",
        "ㅔ":"ㅞ"
        }

    status = ''
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(
            'models/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while cap.isOpened():
        action = ''
        ret, img = cap.read()
        
        if this_action not in ['', ' ']:
            cnt += 1
            jamo_li.append(this_action)
            this_action = ''
            # print(cnt, jamo_li)
            
            status_lst.append(status)
            # print(cnt, status_lst)
            
            if cnt >= status_cnt_conf:
                jamo_dict = {}
                for jamo in jamo_li:
                    jamo_dict[jamo] = jamo_li.count(jamo)
                jamo_dict = Counter(jamo_dict).most_common()
                # print("jamo_dict", jamo_dict)
                if jamo_dict and jamo_dict[0][1] >= int(status_cnt_conf*0.7):
                    tmp = jamo_dict[0][0]
                    status_lst_slice = list(deque(itertools.islice(status_lst, int(status_cnt_conf*0.5), status_cnt_conf-1)))
                    print("status_lst_slice", status_lst_slice)
                    print("tmp", tmp)
                    if tmp in siot:
                        if len(jamo_join_li) == 1:
                            if 'Move' in status_lst_slice: 
                                jamo_join_li.append('ㅆ')
                            else:
                                jamo_join_li.append('ㅅ')
                        else:
                            if jamo_join_li[-1] in J:
                                jamo_join_li.append('ㅠ')
                            else:
                                if 'Move' in status_lst_slice: 
                                    jamo_join_li.append('ㅆ')
                                else:
                                    jamo_join_li.append('ㅅ')
                    elif tmp in J:
                        if tmp in JJ_dict.keys():
                            # 쌍자음
                            if 'Move' in status_lst_slice: 
                                    jamo_join_li.append(JJ_dict[tmp])
                            else:
                                jamo_join_li.append(tmp)
                        else:
                            jamo_join_li.append(tmp)
                    elif tmp in M:
                        # 모음 - 모음 : 이중 모음
                        if jamo_join_li[-1] in MM_lst and tmp in MM_dict.keys():
                            jamo_join_li.pop()
                            jamo_join_li.append(MM_dict[tmp])
                        else:
                            jamo_join_li.append(tmp)
                    elif tmp.isdigit():
                        if len(jamo_join_li) >= 3 and jamo_join_li[-2].isdigit() and jamo_join_li[-1].isdigit():
                            if int(jamo_join_li[-2] + jamo_join_li[-1]) % 10 == 0 and len(tmp) == 1:
                                tmp = str(int(jamo_join_li[-2] + jamo_join_li[-1]) + int(tmp))
                                jamo_join_li.pop()
                                jamo_join_li.pop()
                                for i in tmp:
                                    jamo_join_li.append(i)
                            elif tmp in ["11", "15", "16"]:
                                if jamo_join_li[-2] == "1" and jamo_join_li[-1] == "0":
                                    jamo_join_li.pop()
                                    jamo_join_li.pop()
                                    for i in tmp:
                                        jamo_join_li.append(i)
                            else:
                                for i in tmp:
                                    jamo_join_li.append(i)
                        else:
                            for i in tmp:
                                jamo_join_li.append(i)
                jamo_li = []
                cnt = -int(status_cnt_conf)
                # print("jamo_join_li", jamo_join_li)
                # print("cnt", cnt)

        img, result = detector.findHandswithResult(img, draw=False)
        
        hand_lmlist, _ = detector.findPosition(img, draw=False)

        
        if result.multi_hand_landmarks is not None:
            x1, y1 = hand_lmlist[8][1:3]
            wrist_x, wrist_y = hand_lmlist[0][1:3]
            thumb_index_angle = int(detector.findHandAngle(img, 4, 2, 5, draw=False))
            print(thumb_index_angle)

            # change mode button
            if mode == True:
                if 25 < x1 < 100 and 125 < y1 < 200:
                    mode_count += 1
                    button_overlay = overlayList[2]
                    if mode_count > 15:
                        mode = False
                        mode_count = 0
                        button_overlay = overlayList[1]
                else:
                    button_overlay = overlayList[0]        
            else:
                if 25 < x1 < 100 and 125 < y1 < 200:
                    mode_count +=1
                    button_overlay = overlayList[3]
                    if mode_count > 15:
                        mode = True
                        mode_count = 0
                        button_overlay = overlayList[0]
                else:
                    button_overlay = overlayList[1]
            
            # output delete button
            if 25 < x1 < 100 and 300 < y1 < 375:
                delete_count += 1
                delete_button_overlay = overlayList[5]
                if delete_count > 15:
                    jamo_join_li = deque()
                    s2_lst_remove = ''
                    delete_count = 0
            else:
                delete_count = 0
                delete_button_overlay = overlayList[4]

            # action area
            # cv2.rectangle(img, (100, 100), (540, 400), (255, 255, 255), 1)
            if 100 < x1 < 540 and 100 < y1 < 400:
                # korean mode
                if mode == True:
                    wrist_angle, similar_text_res = wrist_angle_calculator(hand_lmlist)

                    d = vector_normalization(result)

                    seq.append(d)

                    if len(seq) < seq_length:
                        continue

                    input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                    input_data = np.array(input_data, dtype=np.float32)

                    if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
                        # print("model m1")
                        select_model = "m1"
                        i_pred, conf = model_predict(input_data, interpreter_m1)
                        if conf < confidence:
                            continue

                        action = actions_m1[i_pred]

                    elif hand_lmlist[5][1] < hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
                        # print("model m2")
                        select_model = "m2"
                        i_pred, conf = model_predict(input_data, interpreter_m2)
                        
                        if conf < confidence:
                            continue

                        action = actions_m2[i_pred]

                    elif hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[0][2] < hand_lmlist[5][2] and hand_lmlist[0][2] < hand_lmlist[17][2]:
                        # print("model m3")
                        select_model = "m3"
                        i_pred, conf = model_predict(input_data, interpreter_m3)

                        if conf < confidence:
                            continue

                        action = actions_m3[i_pred]
                        if action == 'ㄱ':
                            if thumb_index_angle > 250:
                                action = 'ㅜ'
                        elif action == 'ㅜ':
                            if 35 < thumb_index_angle < 90:
                                action = 'ㄱ'

                    elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2] and (wrist_angle <= 300 or wrist_angle >= 350):
                        # print("model m4")
                        select_model = "m4"
                        i_pred, conf = model_predict(input_data, interpreter_m4)

                        if conf < confidence:
                            continue

                        action = actions_m4[i_pred]
                        # pass
                    
                    elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2]:
                        # print("model m5")
                        select_model = "m5"

                        i_pred, conf = model_predict(input_data, interpreter_m5)

                        if conf < confidence:
                            continue

                        action = actions_m5[i_pred]
                    
                        if action == 'ㄹ':
                            if similar_text_res < 0:
                                action = 'ㅌ'
                            elif 0 < similar_text_res < 20:
                                action = 'ㄹ'

                    
                # Number mode
                else:
                    # x축을 기준으로 손가락 리스트
                    right_hand_fingersUp_list_a0 = detector.fingersUp(axis=False)
                    # y축을 기준으로 손가락 리스트
                    right_hand_fingersUp_list_a1 = detector.fingersUp(axis=True)
                    # 엄지 끝과 검지 끝의 거리 측정
                    thumb_index_length = detector.findLength(4, 8)

                    index_finger_angle_1 = int(detector.findHandAngle(img, 8, 9, 5, draw=False))
                    index_finger_angle_2 = int(detector.findHandAngle(img, 8, 13, 5, draw=False))
                    index_finger_angle_3 = int(detector.findHandAngle(img, 8, 17, 5, draw=False))
                    index_finger_angle_4 = int(detector.findHandAngle(img, 4, 3, 0, draw=False))
                    total_index_angle = index_finger_angle_1 + index_finger_angle_2 + index_finger_angle_3
                    
                    middle_finger_angle_1 = 360 - int(detector.findHandAngle(img, 12, 5, 9, draw=False))
                    middle_finger_angle_2 = int(detector.findHandAngle(img, 12, 13, 9, draw=False))
                    middle_finger_angle_3 = int(detector.findHandAngle(img, 12, 17, 9, draw=False))
                    total_middle_angle = middle_finger_angle_1 + middle_finger_angle_2 + middle_finger_angle_3
                    
                    # 손바닥이 보임, 수향이 위쪽
                    if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[4][2] > hand_lmlist[8][2]:
                        if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmlist[8][2] < hand_lmlist[7][2]:
                            action = 1
                        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 0, 0]:
                            action = 2
                        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 0]:
                            action = 3
                        elif right_hand_fingersUp_list_a0 == [0, 1, 1, 1, 1]:
                            action = 4
                        # elif right_hand_fingersUp_list_a0 == [1, 0, 1, 1, 1] and thumb_index_length < 30:
                        #     action = 10  # 동그라미 10
                        elif thumb_index_length < 30:
                            if right_hand_fingersUp_list_a0 == [1, 0, 1, 1, 1]:
                                action = 10
                            elif right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                                action = 0    
                    # 손바닥이 보임
                    if hand_lmlist[5][1] > hand_lmlist[17][1]:
                        if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                            if right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
                                action = 0
                            else:
                                action = 5
                        # 손가락을 살짝 구부려 10과 20 구분
                        if right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[2:] == [0, 0, 0] and total_index_angle < 140:
                            action = 10
                            cnt10 += 1
                        elif right_hand_fingersUp_list_a0[0] == 0 and right_hand_fingersUp_list_a0[3:] == [0, 0] and total_index_angle < 145 and total_middle_angle < 165:
                            action = 20

                    # 손등이 보임, 수향이 몸 안쪽으로 향함, 엄지가 들려 있음
                    if hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[4][2] < hand_lmlist[8][2]:
                        if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
                            action = 6
                        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 0, 0]:
                            action = 7
                        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 0]:
                            action = 8
                        elif right_hand_fingersUp_list_a1 == [1, 1, 1, 1, 1]:
                            action = 9

                    # 손등이 보이고, 수향이 몸 안쪽으로 향함
                    if hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[1][2] < hand_lmlist[13][2]:
                        # 엄지가 숨어짐
                        if hand_lmlist[4][2] + 30 > hand_lmlist[8][2]:
                            if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 12
                            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 13
                            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 14
                        # 엄지가 보임
                        else:
                            if right_hand_fingersUp_list_a1[2:] == [1, 0, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 17
                            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 0] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 18
                            elif right_hand_fingersUp_list_a1[2:] == [1, 1, 1] and hand_lmlist[8][1] <= hand_lmlist[6][1] + 20:
                                action = 19    

                    if cnt10 > (max_detec - min_detec):
                        action = 10
                        flag = True
                        # print("clear")
                        # dcnt = 0
                        
                        
                    elif cnt10 > min_detec:
                        if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[4][2] > hand_lmlist[8][2]:
                            if right_hand_fingersUp_list_a0 == [0, 1, 0, 0, 0] and hand_lmlist[8][2] < hand_lmlist[7][2]:
                                dcnt += 1
                                action = ''
                                if max_detec > dcnt > min_detec:
                                    action = 11
                                elif dcnt > max_detec+10:
                                    action = 0
                                    cnt10 = 0
                                    dcnt = 0
                                    # print("clear")
                        elif hand_lmlist[5][1] > hand_lmlist[17][1]:
                            if right_hand_fingersUp_list_a0 == [1, 0, 0, 0, 0]:
                                dcnt += 1
                                action = ''
                                if max_detec > dcnt > min_detec:
                                    action = 15
                                elif dcnt > max_detec+10:
                                    action = ''
                                    cnt10 = 0
                                    dcnt = 0
                        elif hand_lmlist[5][2] < hand_lmlist[17][2] and hand_lmlist[4][2] < hand_lmlist[8][2]:
                            if right_hand_fingersUp_list_a1 == [1, 1, 0, 0, 0]:
                                dcnt += 1
                                action = ''
                                if max_detec > dcnt > min_detec:
                                    action = 16
                                elif dcnt > max_detec+10:
                                    action = ''
                                    cnt10 = 0
                                    dcnt = 0
                                    
                        if action in num_lst:
                            flag = True

                    if action != '':
                        if flag:
                            text_cnt += 1
                            if text_cnt % max_detec == 0:
                                cnt10 = 0
                                text_cnt = 0
                                dcnt = 0
                                flag = False

                action = str(action)

                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = ' '
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        last_action = this_action

                # wrist moving check
                status, img = check_moving(result, img, point_history, point_history_classifier, finger_gesture_history, point_history_classifier_labels, draw=False)

            else:
                jamo_li = []
                jamo_join_li.append(' ')
                if jamo_join_li:
                    if len(jamo_join_li) >= 2 and jamo_join_li[-1] == ' ':
                            jamo_join_li.remove(" ")

        img = cv2.flip(img, 1)

        # 자음 모음 결합
        s_lst = list(join_jamos(split_syllables(jamo_join_li)))
        for i in s_lst[:-1]:
            if i in J or i in M:
                s_lst.remove(i)
                jamo_join_li = s_lst
        s2_lst_remove = join_jamos(split_syllables(s_lst))
        
        # Get status box
        cv2.rectangle(img, (0,0), (1000, 60), (245, 117, 16), -1)
        
        # 한글 적용
        b,g,r,a = 255,255,255,0
        # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
        fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
        img_pil = Image.fromarray(img)
        font = ImageFont.truetype(fontpath, 35)
        draw = ImageDraw.Draw(img_pil)
        draw.text((20, 15), f'{this_action}', font=font, fill=(b,g,r,a))
        draw.text((200, 15), f'{s2_lst_remove}', font=font, fill=(b,g,r,a))
        img = np.array(img_pil)

        # Display Probability
        cv2.putText(img, 'INPUT'
                    , (15,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, 'MODEL'
                    , (100,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'{select_model}'
                    , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, 'OUTPUT'
                    , (200,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Display Probability
        cv2.putText(img, 'STATUS'
                    , (550,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, status
        , (550,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img[125:200, 540:615] = button_overlay
        img[300:375, 540:615] = delete_button_overlay

        cv2.rectangle(img, (100, 100), (540, 400), (255, 255, 255), 1)

        cv2.imshow('img', img)

        # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
        if cv2.waitKey(1) & 0xFF == 27:
            break 


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        # landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        ## 좌우이동만 인식하기 위해 y값 제거 ##
        ## 민감도 절감을 위한 x값 조정 ##
        # landmark_point.append([landmark_x, landmark_y])
        landmark_point.append([landmark_x // 1.5, 0])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)

    return image

def model_predict(input_data, interpreter_model):
    interpreter_model.set_tensor(input_details[0]['index'], input_data)
    interpreter_model.invoke()
    y_pred = interpreter_model.get_tensor(output_details[0]['index'])
    i_pred = int(np.argmax(y_pred[0]))
    conf = y_pred[0][i_pred]

    return i_pred, conf

def check_moving(result, img, point_history, point_history_classifier, finger_gesture_history, point_history_classifier_labels, draw=True):
    history_length = 16
    for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
        landmark_list = calc_landmark_list(img, hand_landmarks)

        pre_processed_landmark_list = pre_process_landmark(
            landmark_list)
        pre_processed_point_history_list = pre_process_point_history(
            img, point_history)

        point_history.append(landmark_list[0])

        finger_gesture_id = 0
        point_history_len = len(pre_processed_point_history_list)
        if point_history_len == (history_length * 2):
            finger_gesture_id = point_history_classifier(
                pre_processed_point_history_list)

        # Calculates the gesture IDs in the latest detection
        finger_gesture_history.append(finger_gesture_id)
        most_common_fg_id = Counter(
            finger_gesture_history).most_common()

        status = point_history_classifier_labels[most_common_fg_id[0][0]]
        if draw:
            img = draw_point_history(img, point_history)

        return status, img

def vector_normalization(result):
    for res in result.multi_hand_landmarks:
        joint = np.zeros((21, 2))
        for j, lm in enumerate(res.landmark):
            joint[j] = [lm.x, lm.y]
        
        # Compute angles between joints
        v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
        v = v2 - v1 # [20, 3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

        angle = np.degrees(angle) # Convert radian to degree

        d = np.concatenate([v.flatten(), angle])

        return d

def wrist_angle_calculator(hand_lmlist):
    radian = math.atan2(hand_lmlist[17][2]-hand_lmlist[0][2],hand_lmlist[17][1]-hand_lmlist[0][1])-math.atan2(hand_lmlist[5][2]-hand_lmlist[0][2],hand_lmlist[5][1]-hand_lmlist[0][1])
    wrist_angle = 360 - int(math.degrees(radian))
    radian_2 = math.atan2(hand_lmlist[9][2]-hand_lmlist[12][2],hand_lmlist[9][1]-hand_lmlist[12][1])
    wrist_angle_2 = int(math.degrees(radian_2))
    radian_3 = math.atan2(hand_lmlist[13][2]-hand_lmlist[16][2],hand_lmlist[13][1]-hand_lmlist[16][1])
    wrist_angle_3 = int(math.degrees(radian_3))

    if wrist_angle < 0:
        wrist_angle += 360
        
    if wrist_angle_2 < 0:
        wrist_angle_2 += 360
    if wrist_angle_3 < 0:
        wrist_angle_3 += 360
        
    similar_text_res = wrist_angle_3 - wrist_angle_2

    return wrist_angle, similar_text_res

# User interface variables
mode = True
mode_count = 0
button_overlay = overlayList[0]
delete_count = 0
delete_button_overlay = overlayList[4]

s2_lst_remove = ''

if __name__ == '__main__':
    main(mode, mode_count, button_overlay, delete_count, delete_button_overlay, s2_lst_remove)