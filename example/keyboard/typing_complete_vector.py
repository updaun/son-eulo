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

from hangul_utils import split_syllable_char, split_syllables, join_jamos


# ------------------- 모델 ------------------- #

## m1 방향에 따라 분류(손바닥 위)
# actions_m1 = ['a','ae','ya','yae','i']
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
# model_m1 = load_model('models/M/model_m1.h5')
interpreter_m1 = tf.lite.Interpreter(model_path="models/JM/vector_norm/preprocessing_model1_seq10.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
# actions_m2 = ['o','oe','yo']
actions_m2 = ['ㅇ','ㅗ','ㅚ','ㅛ']
# model_m2 = load_model('models/M/model_m2.h5')
interpreter_m2 = tf.lite.Interpreter(model_path="models/JM/vector_norm/preprocessing_model2_seq10.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
# actions_m3 = ['u','wi','yu']
# actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ','ㅠ']
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ']
# model_m3 = load_model('models/M/model_m3.h5')
interpreter_m3 = tf.lite.Interpreter(model_path="models/JM/vector_norm/preprocessing_model3_seq10.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
# actions_m4 = ['eo','e','yeo','ye']
# actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
actions_m4 = ['ㅎ']
# model_m4 = load_model('models/M/model_m4.h5')
interpreter_m4 = tf.lite.Interpreter(model_path="models/JM/vector_norm/preprocessing_model4_seq10.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
# actions_m5 = ['eu', 'ui']  
# actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅌ','ㅡ','ㅢ']  
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅡ','ㅢ']  
# model_m5 = load_model('models/M/model_m5.h5')
interpreter_m5 = tf.lite.Interpreter(model_path="models/JM/vector_norm/preprocessing_model5_seq10.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()

# -------------------------------------------------- #


def main():
    
    seq_length = 10
    seq = []
    action_seq = []
    last_action = None
    this_action = ''
    select_model = ''

    wrist_angle = 0
    confidence = 0.9

    action = ''
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
    JJ_not_support = ['ㅈ', 'ㅉ', 'ㄷ', 'ㄸ', 'ㅂ', 'ㅃ']
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

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # keypoint_classifier = KeyPointClassifier()

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
                    # print("status_lst_slice", status_lst_slice)
                    # print("tmp", tmp)
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
                
                jamo_li = []
                cnt = -int(status_cnt_conf)
                # print("jamo_join_li", jamo_join_li)
                # print("cnt", cnt)
                
        if not ret:
            break

        h, w, c = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 2))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y]
                    if j == 0:
                        lmlist_0_x, lmlist_0_y = int(lm.x*w), int(lm.y*h)
                    elif j == 4:
                        lmlist_4_x, lmlist_4_y = int(lm.x*w), int(lm.y*h)
                    elif j == 5:
                        lmlist_5_x, lmlist_5_y = int(lm.x*w), int(lm.y*h)
                    elif j == 17:
                        lmlist_17_x, lmlist_17_y = int(lm.x*w), int(lm.y*h)
                        
                    elif j == 12:
                        lmlist_12_x, lmlist_12_y = int(lm.x*w), int(lm.y*h)
                    elif j == 9:
                        lmlist_9_x, lmlist_9_y = int(lm.x*w), int(lm.y*h)
                    elif j == 16:
                        lmlist_16_x, lmlist_16_y = int(lm.x*w), int(lm.y*h)
                    elif j == 13:
                        lmlist_13_x, lmlist_13_y = int(lm.x*w), int(lm.y*h)

                radian = math.atan2(lmlist_17_y-lmlist_0_y,lmlist_17_x-lmlist_0_x)-math.atan2(lmlist_5_y-lmlist_0_y,lmlist_5_x-lmlist_0_x)
                wrist_angle = 360 - int(math.degrees(radian))
                
                radian_2 = math.atan2(lmlist_9_y-lmlist_12_y,lmlist_9_x-lmlist_12_x)
                wrist_angle_2 = int(math.degrees(radian_2))
                radian_3 = math.atan2(lmlist_13_y-lmlist_16_y,lmlist_13_x-lmlist_16_x)
                wrist_angle_3 = int(math.degrees(radian_3))

                if wrist_angle < 0:
                    wrist_angle += 360
                    
                if wrist_angle_2 < 0:
                    wrist_angle_2 += 360
                if wrist_angle_3 < 0:
                    wrist_angle_3 += 360
                    
                similar_text_res = wrist_angle_3 - wrist_angle_2

                # print("wrist_angle : ", wrist_angle)
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

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
                input_data = np.array(input_data, dtype=np.float32)

                if lmlist_5_x > lmlist_17_x and lmlist_5_y < lmlist_0_y and lmlist_17_y < lmlist_0_y:
                    # print("model m1")
                    select_model = "m1"
                    interpreter_m1.set_tensor(input_details[0]['index'], input_data)
                    interpreter_m1.invoke()
                    y_pred = interpreter_m1.get_tensor(output_details[0]['index'])
                    i_pred = int(np.argmax(y_pred[0]))
                    conf = y_pred[0][i_pred]
                    if conf < confidence:
                        continue

                    action = actions_m1[i_pred]

                elif lmlist_5_x < lmlist_17_x and lmlist_5_y < lmlist_0_y and lmlist_17_y < lmlist_0_y:
                    # print("model m2")
                    select_model = "m2"

                    interpreter_m2.set_tensor(input_details[0]['index'], input_data)
                    interpreter_m2.invoke()
                    y_pred = interpreter_m2.get_tensor(output_details[0]['index'])
                    i_pred = int(np.argmax(y_pred[0]))
                    conf = y_pred[0][i_pred]
                    
                    if conf < confidence:
                        continue

                    action = actions_m2[i_pred]

                elif lmlist_5_x > lmlist_17_x and lmlist_0_y < lmlist_5_y and lmlist_0_y < lmlist_17_y:
                    # print("model m3")
                    select_model = "m3"
                    interpreter_m3.set_tensor(input_details[0]['index'], input_data)
                    interpreter_m3.invoke()
                    y_pred = interpreter_m3.get_tensor(output_details[0]['index'])
                    i_pred = int(np.argmax(y_pred[0]))
                    conf = y_pred[0][i_pred]

                    if conf < confidence:
                        continue

                    action = actions_m3[i_pred]
                    if action == 'ㄱ':
                        if lmlist_4_x < lmlist_5_x:
                            action = 'ㅜ'
                    elif action == 'ㅜ':
                        if lmlist_4_x > lmlist_5_x:
                            action = 'ㄱ'

                elif lmlist_5_x > lmlist_0_x and lmlist_5_y < lmlist_17_y and (wrist_angle <= 300 or wrist_angle >= 350):
                    # print("model m4")
                    # select_model = "m4"
                    # interpreter_m4.set_tensor(input_details[0]['index'], input_data)
                    # interpreter_m4.invoke()
                    # y_pred = interpreter_m4.get_tensor(output_details[0]['index'])
                    # i_pred = int(np.argmax(y_pred[0]))
                    # conf = y_pred[0][i_pred]

                    # if conf < confidence:
                    #     continue

                    # action = actions_m4[i_pred]
                    pass
                
                elif lmlist_5_x > lmlist_0_x and lmlist_5_y < lmlist_17_y:
                    # print("model m5")
                    select_model = "m5"
                    interpreter_m5.set_tensor(input_details[0]['index'], input_data)
                    interpreter_m5.invoke()
                    y_pred = interpreter_m5.get_tensor(output_details[0]['index'])
                    i_pred = int(np.argmax(y_pred[0]))
                    conf = y_pred[0][i_pred]

                    if conf < confidence:
                        continue

                    action = actions_m5[i_pred]
                
                    if action == 'ㄹ':
                        if similar_text_res < 0:
                            action = 'ㅌ'
                        elif 0 < similar_text_res < 20:
                            action = 'ㄹ'

                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = ' '
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        last_action = this_action
              
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                landmark_list = calc_landmark_list(img, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    img, point_history)
                
                # logging_csv(number, mode, pre_processed_landmark_list,
                #             pre_processed_point_history_list)

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
                # print(point_history_classifier_labels[most_common_fg_id[0][0]])
                img = draw_point_history(img, point_history)

                # status_lst.append(status)
                # print(cnt, status_lst)
        else:
            jamo_li = []
            jamo_join_li.append(' ')
            if jamo_join_li:
                if len(jamo_join_li) >= 2 and jamo_join_li[-1] == ' ':
                        jamo_join_li.remove(" ")
            # print("jamo_li", jamo_li)

        img = cv2.flip(img, 1)
        
        # 자음 모음 결합
        s2_lst = list(join_jamos(split_syllables(jamo_join_li)))
        for i in s2_lst[:-1]:
            if i in J or i in M:
                s2_lst.remove(i)
                jamo_join_li = s2_lst
        s2_lst_remove = join_jamos(split_syllables(jamo_join_li))
            
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
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

if __name__ == '__main__':
    main()
