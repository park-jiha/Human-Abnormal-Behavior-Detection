#!/usr/bin/env python
# coding: utf-8

# In[1]:


# gpu 확인 -> 1 : 지정됨 / 0 : 지정 안됨

import cv2
rr = cv2.cuda.getCudaEnabledDeviceCount()
print(rr)


# In[1]:


#import cv2
import mediapipe as mp
import numpy as np
import time
import math
from time import time 
#import datetime as dt
#import pandas as pd
#from time import strftime
#from time import gmtime

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

min_confidence = 0.4 # 사람 검출 bounding box 임계값
vedio_path = './220610_test.mp4' # 사용할 영상 경로
t= 0
#-----------------------------------------------------------------------------------
def get_hmsms(dt):
    ms = int((dt - int(dt))*1000)
    dt = int(dt)
    hh, dt = divmod(dt, 3600)
    mm, ss = divmod(dt, 60)
    return (hh, mm, ss, ms)

#print('86399 = ', get_hmsms(86399.334))
#exit(0)
#-----------------------------------------------------------------------------------

def detectAndDisplay(frame):
    # img = cv2.resize(frame, None, fx=0.8, fy=0.8)
    img = frame
    height, width, channels = img.shape

    # 창 크기 설정, input size 지정 가능
    blob = cv2.dnn.blobFromImage(img, 0.00392, (384, 384), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # 탐지한 객체의 클래스 예측 
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 원하는 class id 입력 = 'person' / coco.names의 id에서 -1 할 것 
            if class_id == 0 and confidence > min_confidence:
                # 탐지한 객체 boxing
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
               
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            color = colors[i] #-- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    
    return boxes

# yolo 포맷 및 클래스명 불러오기
# model_file = 'yolov3-spp.weights' # 개발 환경에 맞게 변경할 것
# config_file = 'yolov3-spp.cfg' # 개발 환경에 맞게 변경할 것
model_file = 'yolov4-p6.weights'
config_file = 'yolov4-p6.cfg'
net = cv2.dnn.readNet(model_file, config_file)

# GPU 사용
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# 클래스(names파일) 오픈 / 본인 개발 환경에 맞게 변경할 것
classes = []
with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Video input
cap = cv2.VideoCapture(vedio_path)

if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)

res_actual = np.zeros((1,2), dtype=int)# actual resolution of the camera
res_actual[0,0] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
res_actual[0,1] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print("\ncamera resolution: {}".format(res_actual))

# x, y, w, h = rect
x = int(res_actual[0,1]*0.1)
y = int(res_actual[0,0]*0.1)
w = int(res_actual[0,1]*0.8)
h = int(res_actual[0,0]*0.8)

# 필요 변수 선언
i = 0 # frame count & 저장 리스트 비교용
tmp = 0 # 첫번째 detection 구분
notcnt = 0 # 아무것도 검출이 안되는 상황 대비, 일정 카운트 이상일 시 원본 frame 크기로 변경
case_not = 0 # 검출이 안되다가 갑자기 검출될 시 발생하는 예외 제거
event = 0 # 이벤트 상황 구분
j = 1 # 프레임 별 시간 계산용
check = 1 # 자막 append 간격 지정용
#-----------------------------------------------------------------------------------
FPS = 15
GDt = 1.0/FPS
t1 = 0.0
#-----------------------------------------------------------------------------------

# 프레임 간 좌표 계산을 위한 저장용 list 생성, 초기값 append
PX = []
PY = []
X_min = []
X_max = []
Y_min = []
Y_max = []
X_min.append(x)
X_max.append(x+w)
Y_min.append(y)
Y_max.append(y+h)

PX.append((X_max[i] + X_min[i])/2)
PY.append((Y_max[i] + Y_min[i])/2)

# For xlsx
#from openpyxl import Workbook
#wb = Workbook()
#ws = wb.active


file_srt = open("./220610_test.srt", 'w')

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose: # Mediapipe pose setting
    while cap.isOpened():
        success, img = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          break
        
        # 15 fps 인 경우 setting
        #t1 = j/15
        
        # 첫번째 detection
        if tmp < 1:
            box = detectAndDisplay(img)
            
            #t1 = j/15
            #tsame = t1
            
            # (srt) 시작 time 계산
            #t1 = format(t1, ".5F")
            #start = strftime('%H:%M:%S', gmtime(float(t1)))
            #tmp_sec = (int(start[0])*10+int(start[1]))*3600 + (int(start[3])*10+int(start[4])*60 + (int(start[6])*10)+int(start[7]))
            
            #t1_  = format(float(t1)-tmp_sec, ".5F")
            #t1_ = pd.to_datetime(t1_, format='%S.%f')
            #start_milli = t1_.strftime('%f')            
            #print("start time :", start, ".", start_milli)
            
            # (srt) 종료 time 계산
            #t2 = (j+1)/15
            #t2 = format(t2, ".5F")
            #end = strftime('%H:%M:%S', gmtime(float(t1)))
            #tmp_sec = (int(end[0])*10+int(end[1]))*3600 + (int(end[3])*10+int(end[4])*60 + (int(end[6])*10)+int(end[7]))
            #t2_  = format(float(t2)-tmp_sec, ".5F")
            #t2_ = pd.to_datetime(t2_, format='%S.%f')
            #end_milli = t2_.strftime('%f')
            #print("end time :", end, ".", end_milli)
#-----------------------------------------------------------------------------------
            
            hh1 = 0
            mm1 = 0
            ss1 = 0
            ms1 = 0
            hh2, mm2, ss2, ms2 = get_hmsms(GDt);
 #-----------------------------------------------------------------------------------
           
            
            # 사람이 검출될 시
            if len(box) > 0:
                x = box[0][0]
                y = box[0][1]
                w = box[0][2]
                h = box[0][3]
                notcnt = 0
                if case_not > 0:
                    case_not-=1
            elif len(box) == 0 and i % 10 == 0: # 검출 안될 시 image 크기 조금씩 증가, 오래 검출 안될 시 원본 프레임 크기 복구
                x-=1
                y-=1
                w+=2
                h+=2
                notcnt+=1
                if notcnt % 15 == 0:
                    x = int(res_actual[0,1]*0.1)
                    y = int(res_actual[0,0]*0.1)
                    w = int(res_actual[0,1]*0.8)
                    h = int(res_actual[0,0]*0.8)
                    notcnt = 0
                    case_not = 5

            # 임의로 박스 주변 crop 범위 지정
            x_ = int((2*x-w) / 2)
            y_ = int((2*y-h) / 2)
            w_ = int(2*w)
            h_ = int(2*h)
            xw = x_+w_
            yh = y_+h_

            if x_ <= 0: x_ = 0
            if y_ <= 0: y_ = 0
            if xw >= res_actual[0,1]: xw = res_actual[0,1]
            if yh >= res_actual[0,0]: yh = res_actual[0,0]
            
            ##### 낙상 판단 방향
            X_min.append(x)
            X_max.append(x+w)
            Y_min.append(y)
            Y_max.append(y+h)
            
            cv2.imshow('Falling Detection', img)
            
            if len(box) > 0:
                tmp += 1
#-----------------------------------------------------------------------------------
                
            qq = 'start'
            print("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n" % (j, hh1, mm1, ss1, ms1, hh2, mm2, ss2, ms2, qq), file=file_srt)
#-----------------------------------------------------------------------------------
        
        # crop된 frame 안에서 detection 진행
        elif tmp == 1:
            image = img[y_:yh, x_:xw]
            box = detectAndDisplay(image)
#-----------------------------------------------------------------------------------
####################################################################################            
            t1 = t1 + GDt
            #tsame = t1
            hh1, mm1, ss1, ms1 = get_hmsms(t1)
            hh2, mm2, ss2, ms2 = get_hmsms(t1 + GDt);
#-----------------------------------------------------------------------------------
            
            # (srt) 시작 time 계산
            #t1 = format(t1, ".5F")
            #start = strftime('%H:%M:%S', gmtime(float(t1)))
            #tmp_sec = (int(start[0])*10+int(start[1]))*3600 + (int(start[3])*10+int(start[4])*60 + (int(start[6])*10)+int(start[7]))
            #t1_  = format(float(t1)-tmp_sec, ".5F")
            #t1_ = pd.to_datetime(t1_, format='%S.%f')
            #start_milli = t1_.strftime('%f')
            print("start time : %02d:%02d:%02d,%03d" % (hh1, mm1, ss1, ms1))
            
            # (srt) 종료 time 계산
            #t2 = (j+1)/15
            #t2 = format(t2, ".5F")
            #end = strftime('%H:%M:%S', gmtime(float(t1)))
            #tmp_sec = (int(end[0])*10+int(end[1]))*3600 + (int(end[3])*10+int(end[4])*60 + (int(end[6])*10)+int(end[7]))
            #t2_  = format(float(t2)-tmp_sec, ".5F")
            #t2_ = pd.to_datetime(t2_, format='%S.%f')
            #end_milli = t2_.strftime('%f')
            #print("end time :", end, ".", end_milli)
            print("end time : %02d:%02d:%02d,%03d" % (hh2, mm2, ss2, ms2))
#-----------------------------------------------------------------------------------
            
            # 축소된 프레임 안에서의 좌표 설정, 조정된 값 계산
            if len(box) > 0:
                x = box[0][0] + x_
                y = box[0][1] + y_
                w = box[0][2]
                h = box[0][3]
                notcnt = 0
                if case_not > 0:
                    case_not-=1
            elif len(box) == 0 and i % 10 == 0: # 검출 안될 시 image 크기 조금씩 증가, 오래 검출 안될 시 원본 프레임 크기 복구
                x-=1
                y-=1
                w+=2
                h+=2
                notcnt+=1
                if notcnt % 15 == 0:
                    x = int(res_actual[0,1]*0.1)
                    y = int(res_actual[0,0]*0.1)
                    w = int(res_actual[0,1]*0.8)
                    h = int(res_actual[0,0]*0.8)
                    case_not = 5
            
            # 임의로 박스 주변 crop 범위 지정
            x_ = int((2*x-w) / 2)
            y_ = int((2*y-h) / 2)
            w_ = int(2*w)
            h_ = int(2*h)
            xw = x_+w_
            yh = y_+h_

            if x_ <= 0: x_ = 0
            if y_ <= 0: y_ = 0
            if xw >= res_actual[0,1]: xw = res_actual[0,1]
            if yh >= res_actual[0,0]: yh = res_actual[0,0]
            
            print("box info :", x,y,x+w,y+h)
            
            # 바운딩 박스 좌표 값 저장
            X_min.append(x)
            X_max.append(x+w)
            Y_min.append(y)
            Y_max.append(y+h)
        
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #print
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            ### 낙상 판단 방향
            case_d = 0
            if ((X_max[i] + Y_min[i])/2 > (X_min[i] + Y_max[i])/2) and i>0:
                #좌측낙사 case = 3
                if X_min[i] < X_min[i-1] and X_max[i] < X_max[i-1] and Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                    case_d = 3

                #우측낙사 case = 3
                elif X_min[i] > X_min[i-1] and X_max[i] > X_max[i-1] and Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                    case_d = 3         

                #전방낙사 case = 1
                elif Y_min[i] > Y_min[i-1] and Y_max[i] > Y_max[i-1]:
                    case_d = 1

                #후방낙사 case = 2
                elif Y_min[i] > Y_min[i-1] :
                    case_d = 2

            ### 속력 낙상 판단
            case_v = 0
            PX.append((X_max[i] + X_min[i])/2)
            PY.append((Y_max[i] + Y_min[i])/2)
            Pixel2cm = (Y_max[i-1]+Y_min[i-1])/170

            a = (X_max[i]+X_min[i])/2 - (X_max[i-1]+X_min[i-1])/2
            b = (Y_max[i]+Y_min[i])/2 - (Y_max[i-1]+Y_min[i-1])/2

            #속력 구하기
            V = (math.sqrt(a**2 + b**2) * Pixel2cm)/(1/2)

            if V >= 500 and V < 3000:
                if X_min[i-2] != int(res_actual[0,1]*0.1) and Y_min[i-2] != int(res_actual[0,0]*0.1) and X_max[i-2] != int(res_actual[0,1]*0.9) and Y_max[i-2] != int(res_actual[0,0]*0.9):
                    case_v = 1
            elif V >= 400 and V < 500:
                case_v = 2
            else:
                case_v = 0
            i+=1

            # 33개 keypoint 저장
            if results.pose_landmarks != None:
                landmarks = results.pose_landmarks.landmark
                # 코
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                # 왼쪽 발목
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                # 오른쪽 발목
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                nose[0] = nose[0] * res_actual[0,1]
                nose[1] = nose[1] * res_actual[0,0]
                l_ankle[0] = l_ankle[0] * res_actual[0,1]
                l_ankle[1] = l_ankle[1] * res_actual[0,0]
                r_ankle[0] = r_ankle[0] * res_actual[0,1]
                r_ankle[1] = r_ankle[1] * res_actual[0,0]

                # 1번, 얼굴 좌표(코)의 y값이 하단 좌표(발목 중점)의 y값보다 아래 위치
                # 현재는 사용하지 않음(kepoint 좌표 불안정 문제)
                case_s = 0
                if nose[1] > (l_ankle[1] + r_ankle[1]) / 2:
                    case_s = 1
                else:
                    case_s = 0
            
            # 정상일 때 자막 저장 & 이상 상황 발생 시 event 값 부여
            if event == 0:
                if case_not == 0 and case_d == 1 and case_v == 1: # and case_s == 1:
                    event = 1
                elif case_not == 0 and case_d == 2 and case_v == 1:
                    event = 2
                elif case_not == 0 and case_d == 3 and case_v == 1:
                    event = 3
                elif case_not == 0 and case_d > 0 and case_v == 2:
                    event = 4
                else:
                    print("<<<정상>>>")
                    print("속도 :", V)
                    if check % 15 == 0:
                        #ws.append(['정상', start, start_milli, end, end_milli])
                        qq = '정상'
            
            # 전방 낙상 경우
            elif event == 1:
                print("<<<앞으로 넘어짐!>>>")
                print("속도 :", V)
                if check % 15 == 0:
                    #ws.append(['앞으로 넘어짐!', start, start_milli, end, end_milli])
                    qq = '앞으로 넘어짐!'
            
            # 후방 낙상 경우
            elif event == 2:
                print("<<<뒤로 넘어짐!>>>")
                print("속도 :", V)
                if check % 15 == 0:
                    #ws.append(['뒤로 넘어짐!', start, start_milli, end, end_milli])
                    qq = '뒤로 넘어짐!'
                    
            # 측면 낙상 경우
            elif event == 3:
                print("<<<옆으로 넘어짐!>>>")
                print("속도 :", V)
                if check % 15 == 0:
                    #ws.append(['옆으로 넘어짐!', start, start_milli, end, end_milli])
                    qq = '옆으로 넘어짐!'
            
            # 낙상이 의심되는 상황
            elif event == 4:
                print("<<<낙상 의심>>>")
                print("속도 :", V)
                if check % 15 == 0:
                    #ws.append(['낙상 의심', start, start_milli, end, end_milli])
                    qq = '낙상 의심'
                    event = 0
            
            j+=1
            check+=1
#-----------------------------------------------------------------------------------
            
            print("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n" % (j, hh1, mm1, ss1, ms1, hh2, mm2, ss2, ms2, qq), file=file_srt)
#-----------------------------------------------------------------------------------
            # 결과 show
            cv2.imshow('Falling Detection', image)

            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):
                break

        print("--------------------------------------------------------")
                
#wb.save('log.xlsx') # 로그 저장
file_srt.close()

cap.release()
cv2.destroyAllWindows()


# In[3]:

