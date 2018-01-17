# -*- coding:utf-8 -*-
import cv2
import numpy as np

left_camera_matrix = np.array([[827.01274, 0., 370.00421],
                               [0., 839.64718, 231.39363],
                               [0., 0., 1.]])
left_distortion = np.array([[-0.08616,   2.71576,   -0.01091,   0.02544,  0.00000]])



right_camera_matrix = np.array([[820.13874, 0., 259.77918],
                                [0., 838.88503, 201.85639],
                                [0., 0., 1.]])
right_distortion = np.array([[0.20257,   -0.53978,   -0.04444,   -0.01169,  0.00000]])

om = np.array([-0.09394,   0.10245,  0.01246]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-287.11877,   -8.88822,  -10.95865]) # 平移关系向量

size = (480, 640) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)