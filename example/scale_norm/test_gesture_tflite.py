import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf


# ------------------- 모델 ------------------- #

actions = ['ㅁ','ㅂ','ㅍ','ㅇ','ㅇ','ㅎ','ㅏ','ㅐ','ㅑ','ㅒ','ㅣ']

## j1
# actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', ' ']
# actions = ["giyeok", "shiot", "jieut", "ch'ieuch'", "k'ieuk'"] 
interpreter_j1 = tf.lite.Interpreter(model_path="models/JM/scale_norm/model1.tflite")
interpreter_j1.allocate_tensors()
## j2
# actions = ["nieun", "digeut", "rieul", "t'ieut'"]
# model = load_model('models/J/model_j2.h5')

## j3
# actions = ["mieum", "bieup", "p'ieup'", "ieung_1"]
# model = load_model('models/J/model_j3.h5')

## j4
# actions = ["ieung_2", "hieu"]
# model = load_model('models/J/model_j4.h5')

# -------------------------------------------------- #

# Get input and output tensors.
input_details = interpreter_j1.get_input_details()
output_details = interpreter_j1.get_output_details()

seq_length = 10
confidence = 0.9

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None
this_action = ''

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            x_right_label = []
            y_right_label = []
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
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

            interpreter_j1.set_tensor(input_details[0]['index'], input_data)
            interpreter_j1.invoke()
            y_pred = interpreter_j1.get_tensor(output_details[0]['index'])
            i_pred = int(np.argmax(y_pred[0]))         

            action = actions[i_pred]
            conf = y_pred[0][i_pred]
                
            if conf < confidence:
                continue
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
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
    # 한글 적용
    b,g,r,a = 255,255,255,0
    # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
    fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(fontpath, 35)
    draw = ImageDraw.Draw(img_pil)
    draw.text((30, 15), f'{this_action}', font=font, fill=(b,g,r,a))
    img = np.array(img_pil)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

