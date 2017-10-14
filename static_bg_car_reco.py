#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    Created on Fri Oct 13 17:36:41 CST 2017
    @author: suweitao
"""

import cv2
import numpy as np

class StaticReco(object):
    '''
        @function 初始化变量
        @param equipment 视频设备
    '''
    def __init__(self, equipment):
        self.equipment = equipment
        # 前景获取对象
        self.bs_mog = cv2.createBackgroundSubtractorMOG2()

    '''
        @function 处理视频流设备
    '''
    def readEquipment(self):
        # 读取掉前10帧，启动时不稳定
        for i in xrange(10):
            _, _ = self.equipment.read()

        while(1):
            ret, frame = self.equipment.read()
            frame = cv2.resize(frame, (500, 500))
            # 获取背景
            fgmask = self.bs_mog.apply(frame)

            # 获取轮廓
            _, contours, _ = cv2.findContours(fgmask, 0, 2)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000 and area < 15000:
                    cv2.imshow('fgmask', fgmask)
                    x, y, w, h = cv2.boundingRect(contour)
                    self.shiftVideo(frame, (x, y, w, h))
                    break

        cap.release()


    '''
        @function 视频流中进行图像跟踪
        @param frame 跟踪图像
        @param rect 跟踪图像中要跟踪的位置
    '''
    def shiftVideo(self, frame, rect):
        x, y, w, h = rect
        roi = frame[y:y+h, x:x + w]
        hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 根据亮度和对比度制作掩摸,颜色不处理
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 60.)), np.array((180., 255., 255.)))
        # 计算直方图
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        # 归一化处理，便于比较，不然不能和更大的东西进行比较
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        while (1):
            ret, frame = self.equipment.read()
            if ret == False:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # 直方图反向投影，之前要进行归一化处理
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            # 使用meanshift获取新的位置
            ret, rect = cv2.meanShift(dst, rect, (3L, 10, 1))
            # 绘制正方形
            x, y, w, h = rect
            shift_img = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.imshow('camshift', shift_img)
            if cv2.waitKey(25) & 0xff == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 测试中。。。
    cap = cv2.VideoCapture(0)

    reco = StaticReco(cap)
    reco.readEquipment()

