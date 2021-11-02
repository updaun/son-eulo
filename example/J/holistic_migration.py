import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
import modules.HolisticModule as hm
import numpy as np
from tensorflow.keras.models import load_model


# Holistic 객체 생성
detector = hm.HolisticDetector()

# ------------------- 모델 ------------------- #

## j1
actions = ["giyeok", "shiot", "jieut", "ch'ieuch'", "k'ieuk'"] 
model = load_model('models/J/model_j1.h5')

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

seq_length = 30


# # MediaPipe hands model
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

seq = []
action_seq = []
last_action = None
this_action = ''

while True:

    success, img = cap.read()

    img = cv2.flip(img, 1)
    results, img = detector.findHolisticwithResult(img)
    hand_lmList = detector.findLefthandLandmark(img, draw=True)
    # hand_lmList = detector.findRighthandLandmark(img, draw=True)

    # 감지가 되었는지 확인하는 구문
    if len(hand_lmList) != 0:
        joint = np.zeros((21, 4))
        for j, lm in enumerate(results.left_hand_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

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

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.8:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action

    # Get status box
    cv2.rectangle(img, (0,0), (200, 60), (245, 117, 16), -1)

    # Display Probability
    cv2.putText(img, 'OUTPUT'
                , (15,18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'{this_action.upper()}'
                , (25,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('img', img)
    # ESC 키를 눌렀을 때 창을 모두 종료하는 부분
    if cv2.waitKey(1) & 0xFF == 27:
        break 

