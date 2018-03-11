# -*- coding:utf-8 -*-
import cv2
import numpy as np


left_camera_matrix = np.array([[800.98183, 0., 375.06645],
                               [0., 803.15962, 251.18412],
                               [0., 0., 1.]])
left_distortion = np.array([[0.00629,   1.27180,   0.01658,   0.01335,  0.00000]])



right_camera_matrix = np.array([[790.16560, 0., 329.67330],
                                [0., 790.80848, 216.22939],
                                [0., 0., 1.]])
right_distortion = np.array([[ -0.16366,   1.73946,   -0.00000,   0.00000,  0.00000]])

om = np.array([ -0.08616,   0.03651,  -0.01961]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
# T = np.array([-287.11877,   -8.88822,  -10.95865]) # 平移关系向量
T = np.array([-92.34973,   6.73649,  -8.77558]) # 平移关系向量  -1.33284, -5.10664


size = (640, 480) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
