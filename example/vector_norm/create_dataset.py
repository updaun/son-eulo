import cv2
import mediapipe as mp
import numpy as np
import time, os
from PIL import ImageFont, ImageDraw, Image

actions = ['ㄱ','ㅅ','ㅈ','ㅋ','ㅊ']

seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        # cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        
        # 한글 적용
        b,g,r,a = 255,255,255,0
        # fontpath = "fonts/gulim.ttc" # 30, (30, 25)
        fontpath = "fonts/KoPubWorld Dotum Bold.ttf"
        img_pil = Image.fromarray(img)
        font = ImageFont.truetype(fontpath, 30)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), f'Waiting for collecting {action.upper()} action...', font=font, fill=(b,g,r,a))
        img = np.array(img_pil)
    
        
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            # img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 2)) # 넘파이 배열 크기 변경
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y] # z축 제거, visibility 제거

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0, 9,10,11, 0,13,14,15, 0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize vector
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    # print(v[0])
                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8, 9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([v.flatten(), angle_label])
                    # d = np.concatenate([joint.flatten(), angle_label])
                    print(len(d))
                    print(d)
                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            img = cv2.flip(img, 1)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        # print(action, data.shape)
        # np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        # print(action, full_seq_data.shape)
        # 데이터 저장
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
