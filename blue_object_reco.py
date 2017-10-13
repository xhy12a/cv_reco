# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
    Created on Fri Oct 13 17:36:41 CST 2017
    @author: suweitao
"""

import cv2
import numpy as np

class ColorReco(object):
    def __init__(self):
        pass

    def readVideo(self, equipment):
        while (1):
            # 读取图片
            ret, frame = equipment.read()
            frame = cv2.resize(frame, (500, 500))
            frame = cv2.medianBlur(frame, 5)
            # 转换到 HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 蓝色的阈值(90-120)
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([120, 255, 255])
            # 根据阈值构建掩模
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            image, contours, hierarchy = cv2.findContours(mask, 0, 2)
            for i,contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

            # 显示图像
            cv2.imshow('frame', frame)
            if cv2.waitKey(5) & 0xff == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    reco = ColorReco()

    # 测试中。。。
    cap=cv2.VideoCapture(0)
    reco.readVideo(cap)
