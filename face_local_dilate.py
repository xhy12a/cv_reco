#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import numpy as np

class FaceTool(object):
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)


    '''
        @function 发现人脸，处理，返回处理后的图片
        @param img 内有人脸的图片
        @return 处理后的图像
    '''
    def seeFace(self, img):
        img = cv2.resize(img, (500, 500))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 1.3为查找速度，5为附近邻居数量，发现人脸
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for rect in faces:
            self.localDilate(img, rect)

        # 柔和
        kernel = np.ones((3,3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        return img


    '''
        @function 将图片的局部进行膨胀
        @param img 原图像
        @param rect 要处理的局部
    '''
    def localDilate(self, img, rect):
        (x, y, w, h) = rect
        locate_img = img[y:y+h, x:x+w]
        locate_img = self.localBright(locate_img)

        # 处理
        kernel = np.ones((3, 3), np.uint8)
        locate_img = cv2.erode(locate_img, kernel, iterations=1)
        # 改变原图局部
        img[y:y+h, x:x+w] = locate_img

        cv2.imshow('locate_gray', locate_img)


    '''
        @function 图片提高亮度
        @param img 原图
        @return 亮度提高后的图片
    '''
    def localBright(self, img):
        # 转hsv方便一点
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # hsv_img[:,:,2] += 10 超过部分变黑，所以不用
        h, w = hsv_img.shape[:2]
        for y in xrange(h):
            for x in xrange(w):
                if hsv_img[y, x, 2] <= 245:
                    hsv_img[y, x, 2] += 10

        img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return img


if __name__ == '__main__':
    # 测试。。。
    ft = FaceTool('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    woman_img = cv2.imread('/root/Desktop/woman2.jpg')
    bright_img = ft.seeFace(woman_img)
    cv2.imshow('bright', bright_img)
    cv2.waitKey(0)

    '''
    
    cap = cv2.VideoCapture(0)

    while(1):
        ret, frame = cap.read()
        dilate_img = ft.seeFace(frame)
        cv2.imshow('dilate', dilate_img)

        if cv2.waitKey(25) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    '''
