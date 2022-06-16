# Human-Falling-Detection

## Code Update
- 2022.6.10 ver. : 1차 데모 / model_demo(220610).py
- 2022.6.13 ver. : 1차 데모 후 피드백 -> srt 파일 생성 부분 간소화 / demo_0613.py

## 목적
- 

## Models

### 1. WHENet(https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)
- RGB 이미지에서 오일러 각(Roll, Pitch, Yaw) 값을 계산
- YOLO_v3 를 이용한 Face Detection
- Problem : Version 문제(Tensorflow-gpu)

<img src="https://user-images.githubusercontent.com/62232217/148342110-e2c43c5e-8cb7-4244-b8ca-b97141dce0df.gif"  width="400" height="300"/>
