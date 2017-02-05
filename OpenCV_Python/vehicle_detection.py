# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

cascade_src = 'features/haarcascade_upperbody.xml'
video_src = '../videos/GOPR0195.MP4'
#video_src = '../videos/video1.avi'


#video_src = 'dataset/video2.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    key = cv2.waitKey(1) & 0xFF 
    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()