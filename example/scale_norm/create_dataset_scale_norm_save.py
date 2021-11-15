import cv2
import mediapipe as mp
import numpy as np
import time, os
from PIL import ImageFont, ImageDraw, Image

import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import csv
from scipy import interpolate

actions = ['ㄱ','ㅅ','ㅈ','ㅊ','ㅋ']

seq_length = 30
secs_for_action = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

created_time = int(time.time())

# target1 = list(range(1, 30))
target1 = [5]
target2 = list(range(1, 8))
print("start")
target3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 128, 129]

for videopeople in target1 : # 5
    for videonumber in target2 : # 1, 2, 3, 4, 5, 6, 7
        for videoangle in target3 : # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 128, 129
            videopeople = str(videopeople)
            videoangle = str(videoangle)
            videonumber = str(videonumber)
            print("videopeople:"+ videopeople+" / videoangle:"+ videoangle + " / videonumber:"+ videonumber)
            # 경로 수정 요망
            vidcap = cv2.VideoCapture('.'+videopeople+'/'+videoangle+'/'+videonumber+'.mov')
            
            while(vidcap.isOpened()): 
                ret, img = vidcap.read()
                if not ret: # 새로운 프레임을 못받아 왔을 때 braek
                    break

                data = []

                result = hands.process(img)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 2)) # 넘파이 배열 크기 변경
                        x_right_label = []
                        y_right_label = []
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y] # z축 제거, visibility 제거
                            
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
                        # print(full_scale)
                        # Compute angles between joints
                        v1 = joint[[0,1,2,3,0,5,6,7,0, 9,10,11, 0,13,14,15, 0,17,18,19], :3] # Parent joint
                        v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                        v = v2 - v1 # [20, 3]
                        # Normalize v
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        # print(v[0])
                        # Get angle using arcos of dot product
                        angle = np.arccos(np.einsum('nt,nt->n',
                            v[[0,1,2,4,5,6,8, 9,10,12,13,14,16,17,18],:], 
                            v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                        angle = np.degrees(angle) # Convert radian to degree

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, 0)
                        # print('angle_label', angle_label)
                        print(angle)
                        # d = np.concatenate([joint.flatten(), angle_label])
                        d = np.concatenate([full_scale, angle_label])
                        # print(len(d))
                        # print(d)
                        data.append(d)

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                        
            vidcap.release()
            
        os.makedirs('dataset', exist_ok=True)
        
        data = np.array(data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        # 데이터 저장
        np.save(os.path.join('dataset', f'seq_{videopeople}_{videonumber}_{created_time}'), full_seq_data)
        print("=============================================================================================")