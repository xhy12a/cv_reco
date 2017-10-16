#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    Create on Mon Oct 16 13:13:53 CST 2017

    @author: 苏伟涛
    @version: 1.0
'''

import urllib
import os, sys
from argparse import ArgumentParser
from Queue import Queue
import cv2
import numpy as np


class ImagenetSpider(object):
    def __init__(self, save_path):
        self.urls = Queue(10000)
        if os.path.exists(save_path):
            self.save_path = save_path
        else:
            self.save_path = './cars/'


    '''
        @function 获取下载url
    '''
    def obtainUrl(self, wnid_urls):
        print '[START] obtain url。。。'
        for i in wnid_urls:
            try:
                urls = urllib.urlopen(i).read()
            except Exception as e:
                print str(e)
                continue
            for j in urls.split('\n'):
                self.urls.put(j)

        if not self.urls.empty():
            print '[SUCCESS] obtain url, will retrive.'
        else:
            print '[FAIL] can\'t obtain url'



    '''
        @function 下载图片
    '''
    def retriveImage(self):
        print '[START] retrieve image。。。'
        files = os.listdir(self.save_path)
        file_num = len(files)
        count = file_num
        while not self.urls.empty():
            try:
                down_url = self.urls.get()
                tmp_path = "%s/%d.jpg" % (self.save_path, count)
                urllib.urlretrieve(down_url, tmp_path)
                tmp_img = cv2.imread(tmp_path)
                tmp_img = cv2.resize(tmp_img, (100, 100))
                cv2.imwrite(tmp_path, tmp_img)
                count += 1
            except Exception as e:
                print str(e)


        '''
        	@function 一部分图片下载失败，检测失败图片，删除
        	@param bad_img_path 错误图片的路径，将与之进行匹配
        '''
        def deleteBadImg(self, bad_img_path):
            if not os.path.exists(bad_img_path):
                print '[FAIL] bad_img_path is not exists !!!'

            print '[START] delete bad image。。。'
            # 列出检查路径的所有文件
            files = os.listdir(self.save_path)
            # 加载错误图片的样式，删除所有和该图片一样的图片
            template_img = cv2.imread(bad_img_path)
            delete_num = 0
            for file in files:
                try:
                    now_path = self.save_path + '/' + file
                    check_img = cv2.imread(now_path)

                    if check_img.shape != template_img.shape:
                        check_img = cv2.resize(check_img, template_img.shape)
                    if not np.bitwise_xor(check_img, template_img).any():
                        os.remove(now_path)
                        sys.stdout("[DELETE] delete num:%d\r"%(delete_num))
                        delete_num += 1

                except Exception as e:
                    print str(e)



if __name__ == "__main__":
    '''
    parser = ArgumentParser()
    parser.add_argument('-u', '--url', dest='url', help='must have url')
    parser.add_argument('-o', '--out_path', dest='out_path', default='cars', help='this is save path')

    options = parser.parse_args()
    '''


    # 测试代码。。。
    # 我们需要 5,000 左右的图片去进行训练，来确保正确率
    # 编写时error:本来是想用多线程去下载，然后发现图片有的完整，有的一半，有的无法打开。。。
    spider = ImagenetSpider('/root/Desktop/cars/')

    threads = []
    wnid_urls = []
    wnid_urls.append('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02960352')
    wnid_urls.append('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04285008')
    wnid_urls.append('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02958343')
    wnid_urls.append('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03268790')
    wnid_urls.append('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03079136')

    spider.obtainUrl(wnid_urls)
    spider.retriveImage()
    spider.deleteBadImg('./bad_img.jpg')






