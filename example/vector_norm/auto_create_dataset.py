import cv2
import mediapipe as mp
import numpy as np
import time, os

seq_length = 10
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


created_time = int(time.time())
os.makedirs('dataset3', exist_ok=True)

target1 = list(range(1,41))
target2 = [1,2,4,5]
target3 = [22,24,25,27,28,29,31,33,34,35,37,38,40,41,42,43,44,45,50,53,54,55,58,60,61,62]
for videopeople in target3:
    data = []
    for videoangle in target2:
        for videonumber in target1:
            videopeople = str(videopeople)
            videoangle = str(videoangle)
            videonumber = str(videonumber)
            vidcap = cv2.VideoCapture('C:\\Users\\dropl\\OneDrive\\바탕 화면\\junb\\'+videonumber+'\\'+videoangle+'\\'+videopeople+'.mov')
            print("videopeople:"+ videopeople+" / videoangle:"+ videoangle + " / videonumber:"+ videonumber)

            while vidcap.isOpened():

                ret, img = vidcap.read()
                if not ret: # 새로운 프레임을 못받아 왔을 때 braek
                    break

                img = cv2.flip(img, 1)
            
                
                cv2.imshow('img', img)

                start_time = time.time()

                # while tim
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
                        angle_label = np.append(angle_label,0)

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
    np.save(os.path.join('dataset3', f'seq_{videopeople}_{created_time}'), full_seq_data)
