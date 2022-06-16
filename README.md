# Human Abnormal Behavior Detection

## Code Update
- 2022.6.10 ver. : 1차 데모 / model_demo(220610).py
- 2022.6.13 ver. : 1차 데모 후 피드백 -> srt 파일 생성 부분 간소화 / demo_0613.py

## YOLO cfg, weights
https://github.com/WongKinYiu/ScaledYOLOv4

## 주제 : 이상행동 분류 (낙상 검출) with 싱스웰

## 구성원
### 1. 박지하 / 개발, 리더 / 데이터 수집 및 가공, 학습모델 개발
### 2. 김지은 / 개발 / 데이터 수집 및 가공, 결과 출력 및 저장

## 목적
- 고령층에서 빈번히 일어나는 생활 ‧ 안전 사고에 신속대응과 사전 예방을 목적으로 한다.
- 낙상사고 등으로 인해 골절이나 부상을 입게 될 경우, 직접적으로 거동이 어려워지기 때문에 이로 인한 사고 예방이 필요하다.
- 이상행동 영상데이터 학습을 통해 이상행동(낙상) 등을 분류할 수 있는 모델을 구축한다.
- 본 프로젝트를 통해 이상행동(낙상)을 실시간으로 감지하고, 큰 사고로 이어지는 것을 막아 보호할 수 있는 시스템을 완성하는 것이 목적이다.

## 개발 내용
- 데이터 학습을 통해 사람의 이상행동(낙상)을 감지 및 분류한다.
- Yolo 모델을 이용하여 human detection 후 landmark 추출을 통해 얻은 정보를 바탕으로 낙상 유무를 판단하는 모델 개발도 시도해본다.
- 낙상은 이동 중에 옆으로 낙상, 이동 중에 뒤로 낙상, 이동 중에 앞으로 낙상, 제자리에서 옆으로 낙상, 제자리에서 뒤로 낙상, 제자리에서 앞으로 낙상을 감지하도록 한다.
- Object 좌표 값과 이상행동 유형, 이상행동 시작 프레임, 이상행동 종료 프레임 등을 학습하여 이상행동(낙상)에 따른 행동을 분류한다.
- 여러 모델 개발을 시도하며 최적의 결과를 낼 수 있는 모델로 최종 선정한다.

## Models

### 1. WHENet(https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)
- RGB 이미지에서 오일러 각(Roll, Pitch, Yaw) 값을 계산
- YOLO_v3 를 이용한 Face Detection
- Problem : Version 문제(Tensorflow-gpu)

<img src="https://user-images.githubusercontent.com/62232217/148342110-e2c43c5e-8cb7-4244-b8ca-b97141dce0df.gif"  width="400" height="300"/>
