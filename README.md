# son-eulo


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

## Development Environment
```
python 3.8
```

```
pip install -r requirements.txt
```
