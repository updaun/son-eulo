import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import math
from PIL import ImageFont, ImageDraw, Image
from hangul_utils import split_syllable_char, split_syllables, join_jamos

# ------------------- 모델 ------------------- #

## m1 방향에 따라 분류(손바닥 위)
# actions_m1 = ['a','ae','ya','yae','i']
actions_m1 = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']
# model_m1 = load_model('models/M/model_m1.h5')
interpreter_m1 = tf.lite.Interpreter(model_path="models/JM/jsy_jm_model3.tflite")
interpreter_m1.allocate_tensors()

## m2 방향에 따라 분류(손등 위)
# actions_m2 = ['o','oe','yo']
actions_m2 = ['o','ㅗ','ㅚ','ㅛ']
# model_m2 = load_model('models/M/model_m2.h5')
interpreter_m2 = tf.lite.Interpreter(model_path="models/JM/jsy_jm_model4.tflite")
interpreter_m2.allocate_tensors()

## m3 방향에 따라 분류(아래)
# actions_m3 = ['u','wi','yu']
actions_m3 = ['ㄱ','ㅈ','ㅊ','ㅋ','ㅅ','ㅜ','ㅟ','ㅠ']
# model_m3 = load_model('models/M/model_m3.h5')
interpreter_m3 = tf.lite.Interpreter(model_path="models/JM/jsy_jm_model1.tflite")
interpreter_m3.allocate_tensors()

## m4 방향에 따라 분류 (앞)
# actions_m4 = ['eo','e','yeo','ye']
actions_m4 = ['ㅎ','ㅓ','ㅔ','ㅕ','ㅖ']
# model_m4 = load_model('models/M/model_m4.h5')
interpreter_m4 = tf.lite.Interpreter(model_path="models/JM/jsy_jm_model5.tflite")
interpreter_m4.allocate_tensors()

## m5 방향에 따라 분류 (옆)
# actions_m5 = ['eu', 'ui']  
actions_m5 = ['ㄴ','ㄷ','ㄹ','ㅌ','ㅡ','ㅢ']  
# model_m5 = load_model('models/M/model_m5.h5')
interpreter_m5 = tf.lite.Interpreter(model_path="models/JM/jsy_jm_model2.tflite")
interpreter_m5.allocate_tensors()

# Get input and output tensors.
input_details = interpreter_m1.get_input_details()
output_details = interpreter_m1.get_output_details()


# -------------------------------------------------- #

seq_length = 30


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

seq = []
action_seq = []
last_action = None
this_action = ''
select_model = ''

wrist_angle = 0
confidence = 0.9

cnt = 0
jamo_li = []
jamo_join_li = []

M = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅣ', 'ㅗ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅟ', 'ㅠ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅡ', 'ㅢ']
J = ["ㄱ", "ㅅ", "ㅈ", "ㅊ", "ㅋ", "ㄴ", "ㄷ", "ㄹ", "ㅌ", "ㅁ", "ㅂ", "ㅍ", "ㅇ", "ㅎ", "ㄲ", "ㅆ", "ㅉ", "ㄸ", "ㅃ"]
JJ = ["ㄱ", "ㅅ", "ㅈ", "ㄷ", "ㅂ"]
JJ_dict = {"ㄱ":"ㄲ",
           "ㅅ":"ㅆ",
           "ㅈ":"ㅉ",
           "ㄷ":"ㄸ",
           "ㅂ":"ㅃ"}

while cap.isOpened():
    ret, img = cap.read()
    if not this_action == '':
        cnt += 1
        jamo_li.append(this_action)
        print(cnt, jamo_li)
        if cnt >= 45:
            cnt = 0
            jamo_dict = {}
            for jamo in jamo_li:
                if jamo != ' ' and jamo != '':
                    jamo_dict[jamo] = jamo_li.count(jamo)
            
            jamo_dict = sorted(jamo_dict.items(), key=lambda x:x[1], reverse=True) # (ㄱ, 5), (ㄴ, 4)
            print("jamo_dict", jamo_dict)
            if jamo_dict and jamo_dict[0][1] >= 15:
                tmp = jamo_dict[0][0]
                if jamo_join_li:
                    if tmp in J: # 자음
                        if jamo_join_li[-1] in M: # [-1]모음 - 자음 (=> 받침)
                            jamo_join_li.append(tmp)
                        else: # 자음 - 자음
                            if tmp == jamo_join_li[-1]:
                                if tmp in JJ: # 쌍자음
                                    jamo_join_li.append(JJ_dict[tmp])
                                else: # ex) '안녕'
                                    jamo_join_li.append(tmp)
                            else:
                                if len(jamo_join_li) >= 2: # 받침(뒷)
                                    if jamo_join_li[-2] in M and jamo_join_li[-1] in J:
                                        jamo_join_li.append(tmp)
                    elif tmp in M: # 모음
                        if jamo_join_li[-1] in J:
                            jamo_join_li.append(tmp)
                else: # 맨 처음 시작할 때 자음부터 시작
                    if tmp in J:
                        jamo_join_li.append(tmp)
            
            jamo_li = []
            print("jamo_join_li", jamo_join_li)
            print("cnt", cnt)
            
    if not ret:
        break

    h, w, c = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 2))
            x_right_label = []
            y_right_label = []
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y] # z축 제거, visibility 제거
                if j == 0:
                    lmlist_0_x, lmlist_0_y = int(lm.x*w), int(lm.y*h)
                elif j == 5:
                    lmlist_5_x, lmlist_5_y = int(lm.x*w), int(lm.y*h)
                elif j == 17:
                    lmlist_17_x, lmlist_17_y = int(lm.x*w), int(lm.y*h)
            
            radian = math.atan2(lmlist_17_y-lmlist_0_y,lmlist_17_x-lmlist_0_x)-math.atan2(lmlist_5_y-lmlist_0_y,lmlist_5_x-lmlist_0_x)
            wrist_angle = 360 - int(math.degrees(radian))

            if wrist_angle < 0:
                wrist_angle += 360
            # print("wrist_angle : ",wrist_angle)
            for i in range(21):
                x_right_label.append(joint[i][0] - joint[0][0])
                y_right_label.append(joint[i][1] - joint[0][1])

            if max(x_right_label) == min(x_right_label):
                x_right_scale = x_right_label
            else:
                x_right_scale = x_right_label/(max(x_right_label)-min(x_right_label))
                
            if max(y_right_label) == min(y_right_label):
                y_right_scale = y_right_label
            else:
                y_right_scale = y_right_label/(max(y_right_label)-min(y_right_label))
            full_scale = np.concatenate([x_right_scale.flatten(), y_right_scale.flatten()])

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

            d = np.concatenate([full_scale, angle])

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

            elif lmlist_5_x > lmlist_0_x and lmlist_5_y < lmlist_17_y and (wrist_angle <= 300 or wrist_angle >= 350):
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
            
            # print(conf)

            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = ' '
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

                if last_action != this_action:
                    last_action = this_action
                
    img = cv2.flip(img, 1)

    st = split_syllables(jamo_join_li)
    # print(st)
    s2 = join_jamos(st)
    # print(s2)

    # Get status box
    cv2.rectangle(img, (0,0), (1000, 60), (245, 117, 16), -1)

    # Display Probability
    cv2.putText(img, 'INPUT'
                , (15,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 한글 적용
    b,g,r,a = 255,255,255,0
    # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
    fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(fontpath, 35)
    draw = ImageDraw.Draw(img_pil)
    draw.text((20, 15), f'{this_action}', font=font, fill=(b,g,r,a))
    draw.text((200, 15), f'{s2}', font=font, fill=(b,g,r,a))
    img = np.array(img_pil)

    # Display Probability
    cv2.putText(img, 'MODEL'
                , (100,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'{select_model}'
                , (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, 'OUTPUT'
                , (200,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # cv2.putText(img, 'COUNT'
    #             , (500,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, f'{cnt}'
    #             , (500,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('img', img)
    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

