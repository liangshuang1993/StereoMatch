# -*- coding:utf-8 -*-
import numpy as np
import cv2
import camera_configs

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("sigma", "depth", 1, 20, lambda x: None)
cv2.createTrackbar("lmbda", "depth", 1, 10, lambda x: None)
cv2.createTrackbar("visual_multiplier", "depth", 5, 20, lambda x: None)

threeD = []

def get_depth_map(image1, image2):
    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(image1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(image2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    # imgL = img1_rectified
    # imgR = img2_rectified


    cv2.imshow('left', image1)
    cv2.imshow('right', image2)
    # 两个trackbar用来调节不同的参数查看效果
    visual_multiplier = cv2.getTrackbarPos("visual_multiplier", "depth") / 10.0
    sigma = cv2.getTrackbarPos("sigma", "depth") / 10.0
    lmbda = cv2.getTrackbarPos("lmbda", "depth") * 10000

    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    #lmbda = 80000
    #sigma = 1.2
    #visual_multiplier = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None,
                                    dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255,
                                norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    threeD = cv2.reprojectImageTo3D(filteredImg, camera_configs.Q)

    return threeD

def get_depth_by_rect(image1, image2, rect):
    threeD = get_depth_map(image1, image2)

    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera1 = cv2.VideoCapture(1)
    camera2 = cv2.VideoCapture(2)
    while True:
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        # object detection
        rect = detection()
        # get depth
        get_depth_by_rect(frame1, frame2, rect)