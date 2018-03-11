# -*- coding:utf-8 -*-  
import cv2
import numpy as np


cap1 = cv2.VideoCapture(2)
#cap2 = cv2.VideoCapture(2)
fourcc = cv2.cv.CV_FOURCC(*'XVID')
#opencv3的话用:fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))#保存视频
while True:
    ret1,frame1 = cap1.read()
#    ret2,frame2 = cap2.read()
    # rotate 180
#    rows,cols = frame2.shape[:2]
    #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
#    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
    #第三个参数：变换后的图像大小
#    frame2 = cv2.warpAffine(frame2,M,(cols,rows))

#    gray1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#    gray2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    #out.write(frame)#写入视频
    cv2.imshow('frame1',frame1)#一个窗口用以显示原视频
    #cv2.imshow('gray1',gray1)#另一窗口显示处理视频
#    cv2.imshow('frame2',frame2)#一个窗口用以显示原视频
    #cv2.imshow('gray2',gray2)#另一窗口显示处理视频


    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cap1.release()
#out.release()
#cap2.release()
cv2.destroyAllWindows()
