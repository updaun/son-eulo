import sys
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from pynput.keyboard import Key, Controller

kbControl = Controller()

# actions = ['a','ae','ya','yae','i']    # 방향에 따라 분류(손바닥 위)
# model = load_model('models/M/model_m1.h5')

# actions = ['o','oe','yo']              # 방향에 따라 분류(손등 위)
# model = load_model('models/M/model_m2.h5')

# actions = ['u','wi','yu']              # 방향에 따라 분류(아래)
# model = load_model('models/M/model_m3.h5')

# actions = ['eo','e','yeo','ye']        # 방향에 따라 분류 (앞)
# model = load_model('models/M/model_m4.h5')

actions = ['eu', 'ui']  # 방향에 따라 분류 (옆)
model = load_model('models/M/model_m5.h5')

seq_length = 30

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

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
            v = v2 - v1  # [20, 3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            angle = np.degrees(angle)  # Convert radian to degree

            d = np.concatenate([joint.flatten(), angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            y_pred = model.predict(input_data).squeeze()

            i_pred = int(np.argmax(y_pred))
            conf = y_pred[i_pred]

            if conf < 0.5:
                continue

            action = actions[i_pred]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = action
            # this_action = '?'
            # if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            #     this_action = action

            #     if last_action != this_action:
            #         if this_action == 'go':

            #             kbControl.press(Key.right)
            #             kbControl.release(Key.right)

            #         elif this_action == 'back':

            #             kbControl.press(Key.left)
            #             kbControl.release(Key.left)

            #         elif this_action == 'start':

            #             kbControl.press(Key.f5)
            #             kbControl.release(Key.f5)

            #         elif this_action == 'finish':

            #             kbControl.press(Key.esc)
            #             kbControl.release(Key.esc)

            #         last_action = this_action
            ####################################################
            # text input test
            ####################################################
            # if last_action != this_action:
            #     if this_action == 'go':

            #         kbControl.press('g')
            #         kbControl.release('g')

            #     elif this_action == 'back':

            #         kbControl.press('b')
            #         kbControl.release('b')

            #     elif this_action == 'start':

            #         kbControl.press('s')
            #         kbControl.release('s')

            #     elif this_action == 'finish':

            #         kbControl.press('f')
            #         kbControl.release('f')

            #     last_action = this_action

            cv2.putText(img, f'{this_action.upper()}',
                        org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

