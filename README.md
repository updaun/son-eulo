## 지화 모델 구축(손으로 프로젝트)
<hr>

![image](https://user-images.githubusercontent.com/82289435/176120587-b2924946-a77e-4471-ae1b-ef0edb0ccad0.png)

![image](https://user-images.githubusercontent.com/82289435/176120671-98cd1bad-7812-468c-88ef-b824577d5138.png)

![image](https://user-images.githubusercontent.com/82289435/176120825-a44314c7-7008-4a64-a1af-91bfb520ccdb.png)

![image](https://user-images.githubusercontent.com/82289435/176120902-07187ce2-6c67-46b1-ad79-d11ea592839e.png)

![image](https://user-images.githubusercontent.com/82289435/176120995-b74ca2ca-917d-4c35-b35a-5e7e9c228916.png)

![image](https://user-images.githubusercontent.com/82289435/176121065-c6cc1424-26bf-44c3-9a04-49e3f6105a5c.png)

![image](https://user-images.githubusercontent.com/82289435/176121125-ca637ed3-f9d2-45a8-a19e-82dc919cc354.png)

![image](https://user-images.githubusercontent.com/82289435/176121223-9e91ae62-5950-4b38-b29e-316e57a9411f.png)

![image](https://user-images.githubusercontent.com/82289435/176121293-fa209fd0-9aba-4be4-b060-7b9da53c5be3.png)

![image](https://user-images.githubusercontent.com/82289435/176121358-bb01a9e2-0788-47ea-acb0-928b9ab75328.png)

## 기업 프로젝트 성과 발표
<hr>

#### 광주인공지능사관학교 졸업식 발표 영상
#### 이미지를 클릭하면 유튜브 영상이 나옵니다.
[![Video Label](https://user-images.githubusercontent.com/82289435/176122132-5350eab2-2edd-4ac0-80c2-0581d7ba9228.png)](https://www.youtube.com/watch?v=vyCmKMl3398&t=5975s)

## 시연 영상
<hr>

### 🚩 주소입력 (첨단과기로339)

![test1](https://user-images.githubusercontent.com/82289435/177174951-a2794feb-9e2d-4456-bb9e-31415ee4d51f.gif)

### 🎓 교장샘 싸랑해요 💖

![user_test_daun](https://user-images.githubusercontent.com/82289435/177180598-ed120ae0-6925-45b0-a847-0f0ff8586e4e.gif)

### ☕ 아메리카노

![test3](https://user-images.githubusercontent.com/82289435/177180314-f407150c-fd18-4359-88c0-1bfcb5709090.gif)

### 🥃 카페라떼 

![test4](https://user-images.githubusercontent.com/82289435/177180415-7e6db02d-66a3-4fad-b9f4-9ef2cbb9e34a.gif)



## Directory Structure

```
.
├── README.md
├── requirements.txt 
├── son-eulo_app.py 
├── examples
│   ├── check_moving : 손목 움직임 감지
│   │    └── check_moving_test.py
│   ├── integration : UI 통합
│   │    └── integration_demo_complete.py
│   ├── J : 자음
│   │    ├── hand_migration_h5.py
│   │    └── hand_migration_tflite.py
│   ├── M : 모음
│   │    ├── hand_migration_h5.py
│   │    └── hand_migration_tflite.py
│   ├── JM : 자음 + 모음
│   │    ├── scale_migration_tflite.py
│   │    └── vector_migration_tflite.py
│   ├── keyboard : 키보드 입력
│   │    ├── hangul_utils_test.py
│   │    └── typing_complete_vector.py
│   ├── similar_text_detection : "ㄹ", "ㅌ" 구분
│   │    └── similar_text_detection.py
│   ├── KSLN : 지숫자
│   │    └── hand_numcount_complete.py
│   ├── scale_norm : 좌표정규화
│   │    ├── auto_create_dataset.py
│   │    ├── cam_create_dataset.py
│   │    ├── test_gesture_tflite.py
│   │    └── test_gesture.py
│   ├── vector_norm : 벡터정규화
│   │    ├── auto_create_dataset.py
│   │    ├── cam_create_dataset.py
│   │    ├── test_gesture_tflite.py
│   └──  └── test_gesture.py
├── models
│   ├── J : 자음 
│   ├── M : 모음
│   ├── JM : 자음 + 모음
│   │    ├── scale_norm : 좌표 정규화
│   │    └── vector_norm : 벡터 정규화
│   └── point_history_classifier
├── fonts
├── images/button_image : UI 버튼 이미지
├── modules : Mediapipe Class Modules
│   ├── HandTrackingModules.py
└── └── HolisticModule.py
```

## Get Started
```
python 3.8
```

```
pip install -r requirements.txt
```

```
python son-eulo_app.py
```

## 시연 영상(숫자)
<hr>

### 숫자 0~5

![1to5](https://user-images.githubusercontent.com/82289435/177180854-cec0a51d-c841-4354-847c-e1f6ee809324.gif)

### 숫자 6~10

![6to10](https://user-images.githubusercontent.com/82289435/177180927-861dbb07-5b98-4747-a2e5-39b6db7f615c.gif)

### 숫자 11~12

![11to12](https://user-images.githubusercontent.com/82289435/177180964-b59138e6-d543-494b-906b-e1a870ae8f37.gif)

### 숫자 13~16

![13to16](https://user-images.githubusercontent.com/82289435/177181012-6a9d8566-2274-4878-96cd-1747ecb2a834.gif)

### 숫자 17~20

![17to20](https://user-images.githubusercontent.com/82289435/177181081-d096b68b-fc5a-42ff-858c-22c908d18fa2.gif)