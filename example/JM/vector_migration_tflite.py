import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from PIL import ImageFont, ImageDraw, Image

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import modules.HandTrackingModule as htm

# Hand 객체 생성
detector = htm.handDetector(max_num_hands=1)

## m1 방향에 따라 분류(손바닥 위)
# actions_m1 = ['a','ae','ya','yae','i']
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅇ','ㅎ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
# model_m1 = load_model('models/M/model_m1.h5')
interpreter_m1 = tf.lite.Interpreter(model_path="models\modelvector\model1.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
# actions_m2 = ['o','oe','yo']
actions_m2 = ['ㅇ','ㅎ','ㅗ','ㅚ','ㅛ']
# model_m2 = load_model('models/M/model_m2.h5')
interpreter_m2 = tf.lite.Interpreter(model_path="models\modelvector\model2.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
# actions_m3 = ['u','wi','yu']
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ']
# model_m3 = load_model('models/M/model_m3.h5')
interpreter_m3 = tf.lite.Interpreter(model_path="models\modelvector\model3.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
# actions_m4 = ['eo','e','yeo','ye']
actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
# model_m4 = load_model('models/M/model_m4.h5')
interpreter_m4 = tf.lite.Interpreter(model_path="models\modelvector\model4.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
# actions_m5 = ['eu', 'ui']  
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅡ','ㅢ']  
# model_m5 = load_model('models/M/model_m5.h5')
interpreter_m5 = tf.lite.Interpreter(model_path="models\modelvector\model5.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()


# -------------------------------------------------- #

seq_length = 10


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture('C:\\Users\\dropl\\OneDrive\\바탕 화면\\accvideo\\da.mp4')

seq = []
action_seq = []
last_action = None
this_action = ''
select_model = ''

wrist_angle = 0
confidence = 0.9
action = ' '

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, c = img.shape

    # img = cv2.flip(img, 1)
    
    img, result = detector.findHandswithResult(img, draw=False)

    hand_lmlist, _ = detector.findPosition(img, draw=False)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 2))

            thumb_index_angle = int(detector.findHandAngle(img, 4, 2, 5, draw=False))

            radian = math.atan2(hand_lmlist[17][2]-hand_lmlist[0][2],hand_lmlist[17][1]-hand_lmlist[0][1])-math.atan2(hand_lmlist[5][2]-hand_lmlist[0][2],hand_lmlist[5][1]-hand_lmlist[0][1])
            wrist_angle = 360 - int(math.degrees(radian))

            if wrist_angle < 0:
                wrist_angle += 360

            print("wrist_angle : ", wrist_angle)
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

            if hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
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

            elif hand_lmlist[5][1] < hand_lmlist[17][1] and hand_lmlist[5][2] < hand_lmlist[0][2] and hand_lmlist[17][2] < hand_lmlist[0][2]:
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

            elif hand_lmlist[5][1] > hand_lmlist[17][1] and hand_lmlist[0][2] < hand_lmlist[5][2] and hand_lmlist[0][2] < hand_lmlist[17][2]:
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
                    if thumb_index_angle > 250:
                        action = 'ㅜ'
                elif action == 'ㅜ':
                    if 35 < thumb_index_angle < 90:
                        action = 'ㄱ'

            elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2] and (wrist_angle <= 300 or wrist_angle >= 350):
                # print("model m4")
                select_model = "m4"
                interpreter_m4.set_tensor(input_details[0]['index'], input_data)
                interpreter_m4.invoke()
                y_pred = interpreter_m4.get_tensor(output_details[0]['index'])
                i_pred = int(np.argmax(y_pred[0]))
                conf = y_pred[0][i_pred]

                if conf < confidence:
                    continue

                action = actions_m4[i_pred]

            elif hand_lmlist[5][1] > hand_lmlist[0][1] and hand_lmlist[5][2] < hand_lmlist[17][2]:
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
            
            

            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = ' '
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

                if last_action != this_action:
                    last_action = this_action
                
    img = cv2.flip(img, 1)

    # Get status box
    cv2.rectangle(img, (0,0), (260, 60), (245, 117, 16), -1)

    # Display Probability
    cv2.putText(img, 'OUTPUT'
                , (15,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, f'{this_action.upper()}'
    #             , (15,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 한글 적용
    b,g,r,a = 255,255,255,0
    # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
    fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(fontpath, 35)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 15), f'{this_action}', font=font, fill=(b,g,r,a))
    img = np.array(img_pil)

    # Display Probability
    cv2.putText(img, 'MODEL'
                , (200,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'{select_model}'
                , (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('img', img)
    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 