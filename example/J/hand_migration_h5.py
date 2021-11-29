import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

# ------------------- 모델 ------------------- #

## j1
# actions_j1 = ["giyeok", "shiot", "jieut", "ch'ieuch'", "k'ieuk'"] 
actions_j1 = ["ㄱ", "ㅅ", "ㅈ", "ㅊ", "ㅋ"] 
model_j1 = load_model('models/J/model_j1.h5')

## j2
# actions_j2 = ["nieun", "digeut", "rieul", "t'ieut'"]
actions_j2 = ["ㄴ", "ㄷ", "ㄹ", "ㅌ"]
model_j2 = load_model('models/J/model_j2.h5')

## j3
# actions_j3 = ["mieum", "bieup", "p'ieup'", "ieung_1"]
actions_j3 = ["ㅁ", "ㅂ", "ㅍ", "ㅇ_1"]
model_j3 = load_model('models/J/model_j3.h5')

## j4
# actions_j4 = ["ieung_2", "hieu"]
actions_j4 = ["ㅇ_2", "ㅎ"]
model_j4 = load_model('models/J/model_j4.h5')

# -------------------------------------------------- #

seq_length = 30
confidence = 0.6


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

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    h, w, c = img.shape

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                if j == 0:
                    lmlist_0_x, lmlist_0_y = int(lm.x*w), int(lm.y*h)
                elif j == 5:
                    lmlist_5_x, lmlist_5_y = int(lm.x*w), int(lm.y*h)
                elif j == 17:
                    lmlist_17_x, lmlist_17_y = int(lm.x*w), int(lm.y*h)
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

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            if lmlist_5_x < lmlist_17_x and lmlist_5_y < lmlist_0_y and lmlist_17_y < lmlist_0_y:
                print("model j3")
                select_model = "j3"
                y_pred = model_j3.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]
                
                if conf < confidence:
                    continue

                action = actions_j3[i_pred]

            elif lmlist_5_x < lmlist_17_x and lmlist_0_y < lmlist_5_y and lmlist_0_y < lmlist_17_y:
                print("model j1")
                select_model = "j1"
                y_pred = model_j1.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < confidence:
                    continue

                action = actions_j1[i_pred]

            elif lmlist_5_x < lmlist_0_x and lmlist_5_y < lmlist_17_y:
                print("model j2")
                select_model = "j2"
                y_pred = model_j2.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < confidence:
                    continue

                action = actions_j2[i_pred]
            else:
                print("model j4")
                select_model = "j4"
                y_pred = model_j4.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < confidence:
                    continue

                action = actions_j4[i_pred]
            

            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action

                if last_action != this_action:
                    last_action = this_action
                

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

