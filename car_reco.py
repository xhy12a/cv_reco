#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
	Created on Fri Oct  6 17:24:24 CST 2017

	@version: 1.0
	@author: suweitao
"""

import numpy as np
import cv2, os
import urllib


class dynamicReco(object):
	def __init__(self):
		# self.svm = svm = cv2.ml.SVM_create()
		pass

	"""
	'''
		@function 读取图片，用k类聚将图片颜色量化
		@param img_path 要读取的图片的路径
		@return 返回量化图片object
	'''
	def quantificationColor(self, img_path):
		img = cv2.imread(img_path)
		low = img.reshape((-1,3)).astype('float32')
		ret, label, center = cv2.kmeans(low, 8, (3L, 10, 1.0), 10, 0)
		center = np.uint8(center)
		ori = center[label.ravel()].reshape(img.shape)
		return ori


	'''
		@function 计算特征向量
		@param img 一张图片
		@return 64特征
	'''
	def hog(self, img):
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bins = np.int32(16*ang/(2*np.pi))
		bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)
		return hist


	'''
		@function 使用svm进行机器学习
		@param real_path 正确图片的路径
		@param false_path 错误图片的路径
		@param label 正确图片的标签
		@param save_path svm参数储存的路径
	'''
	def trainSVMClassifier(self, real_path, false_path, label, save_path):
		if not os.path.exists(real_path) and not os.path.exists(false_path):
			print '[ERROR] path is not exists !!!'
			return

		hogData = []
		real_files = os.listdir(real_path)
		false_files = os.listdir(false_path)
		real_len = len(real_files)
		false_len = len(false_files)

		print "[START] read file ..."
		for i in real_files:
			# hogData.append(self.hog(self.quantificationColor("%s/%s"%(learn_path, i))))
			hogData.append(self.hog(cv2.imread("%s/%s"%(real_path, i))))
		for i in false_files:
			hogData.append(self.hog(cv2.imread("%s/%s"%(false_path, i))))

		trainData = np.float32(hogData).reshape(-1,64)
		print '[SUCCESS] trainData is ready.'
		labels = np.hstack((np.repeat(sum([ord(i) for i in label]), real_len), np.repeat(np.array([-1]), false_len)))		
		responses = np.float32(labels[:,np.newaxis])
		self.svm.train(trainData, responses, params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 ))
		print '[END] SVM train is end.'
		self.svm.save(save_path)


	'''
		@function 使用svm对图片进行检测 
		@param real_path 测试图片的路径
		@return 返回测试结果
	'''
	def runSVMClassifier(self, real_path):
		if not os.path.exists(real_path):
			print '[ERROR] path is not exists !!!'
			return

		hogData = []
		real_files = os.listdir(real_path)
		real_len = len(real_files)

		print "[NOW] read file.\n"
		for i in real_files:
			# hogData.append(self.hog(self.quantificationColor("%s/%s"%(learn_path, i))))
			hogData.append(self.hog(cv2.imread("%s/%s"%(real_path, i))))

		testData = np.float32(hogData).reshape(-1,64)
		print '[SUCCESS] trainData is ready.'
		responses = np.float32(np.repeat(sum([ord(i) for i in label]), real_len)[:,np.newaxis])

		###### Check Accuracy #######
		result = self.svm.predict_all(testData)
		return result
	"""

	'''
		@function 视频拆帧
		@param video_path 视频路径
		@param save_path 帧图片保存路径
	'''
	def videoToImg(self, video_path, save_path):
		cap = cv2.VideoCapture(video_path)
		if not os.path.exists(video_path):
			print '[ERROR] video_path is not exists !!!'
		elif not os.path.exists(save_path):
			os.mkdir(save_path)

		count = 1
		while True:
			ret, img = cap.read()
			if not ret:
				break 
			img = cv2.resize(img, (200,200))
			cv2.imwrite("%s/%d.jpg"%(save_path, count), img)
			print '[NOW] write %d.jpg'%(count,)
			count += 1


	'''
		@function 爬虫
		@param download_url image-net.org的download_url
		@param save_path 图片的储存路径
	'''
	def store_raw_images(self, download_url, save_path):
		# 判断参数有效性
		if not os.path.exists(save_path):
			os.mkdir(save_path)

		# 列出保存路径中已经存在的文件,
		before_files = os.listdir(save_path)

		# 开始爬取图片，用于后面的训练
		try:
			neg_image_urls = urllib.urlopen(download_url).read()
			pic_num = len(before_files)+1
			for i in neg_image_urls.split('\n'):
				print i
				try:
					tmp_path = "%s/%d.jpg"%(save_path, pic_num)
					# 下载图片到路径中
					urllib.urlretrieve(i, tmp_path)
					# 读取，并且更改大小,然后写入
					img = cv2.imread(tmp_path)
					resized_img = cv2.resize(img, (100, 100))
					cv2.imwrite(tmp_path, resized_img)
					pic_num += 1
				except Exception as ee:
					print str(ee)

		except Exception as e:
			print str(e)


	'''
		@function 一部分图片下载失败，检测失败图片，删除
		@param check_path 要进行检查的路径
	'''
	def deleteBadImg(self, check_path):
		# 判断参数有效性
		if not os.path.exists(check_path):
			print '[ERROR] the path is not exists !!!'
			return

		# 列出检查路径的所有文件
		files = os.listdir(check_path)
		# 加载错误图片的样式，删除所有和该图片一样的图片
		template_img = cv2.imread('./bad_load_img.jpg')
		for file in files:
			try:
				now_path = check_path+'/'+file
				check_img = cv2.imread(now_path)

				if check_img.shape != template_img.shape:
					check_img = cv2.resize(check_img, template_img.shape)
				if not np.bitwise_xor(check_img, template_img).any():
					os.remove(now_path)
					print '[COMPLETE] delete a', now_path

			except Exception as e:
				print str(e)


	'''
		@function 创建bg.txt(储存所有训练图片路径),使用opencv_createsamples会用到
		@param path 训练图片文件夹
	'''
	def createPathFile(self, path):
		# 检查文件夹路径是否存在
		if not os.path.exists(path):
			print '[ERROR] the path is not exists !!!'
			return

		# 批量把文件夹中的文件的路径写入
		files = os.listdir(path)
		with open('bg.txt', 'w') as f:
			for i in files:
				line = path+'/'+i+'\n'
				f.write(line)
		print '[SUCCESS] good write path to file.'


	'''
		@function 使用opencv_createsamples和opencv_traincascade进行神经网络训练,得到cascade.xml
		@param template_img 标志性图片，作为第一张图片，用于info.lst扣图
		@param bg_path bg.txt的路径
		@param imgs_path 训练图片的路径
	'''
	def callOpencvSample(self, template_img, bg_path, imgs_path):
		# 判断图片路径是否存在，并且创建data和info文件夹
		if not os.path.exists(template_img) or not os.path.exists(bg_path) or not os.path.exists(imgs_path):
			print '[ERROR] path have not !!!'
		if not os.path.exists('data'):
			os.system('mkdir data')
		if not os.path.exists('info'):
			os.system('mkdir info')
		
		# 根据文件夹中文件数量得到num,作为后面命令的参数
		num = len(os.listdir(imgs_path))
		num_pos = num - (num%100)
		if (num_pos/100)%2 == 1:
			num_pos -= 100

		# 训练神经网络,得到xml
		print '[SUCCESS] great to info.lst !!!\n', os.popen("opencv_createsamples -img %s -bg %s -info info/info.lst -pngoutput info -maxxangle 0.5 -maxyangle 0.5 -num %d"%(template_img, bg_path, num)).read()
		print '[SUCCESS] great to positives.vec !!!\n', os.popen("opencv_createsamples -info info/info.lst -num %d -w 20 -h 20 -vec positives.vec"%(num, )).read()
		print '[SUCCESS] great to train !!!\n', os.popen("opencv_traincascade -data data -vec positives.vec -bg %s -numPos %d -numNeg %d -numStages 10 -w 20 -h 20"%(bg_path, num_pos, num_pos/2))


	'''
		@function 使用cascade进行识别，判断初步位置,然后用camshift进行跟踪
		@param video_equipment 视频设备，可以是视频，也可以是摄像头
		@param xml_path xml文件的路径
	'''
	def drawFeaturesForVideo(self, video_equipment, xml_path):
		# 判断参数合法性
		if not os.path.exists(xml_path):
			print '[ERROR] xml is not exists !!!'
		tmp_equipment = str(video_equipment)
		if not os.path.exists(tmp_equipment) and not tmp_equipment in ['0', '1']:
			print '[ERROR] video_equipment is bad !!!'

		# 加载视频流，加载串联分类器
		cap = cv2.VideoCapture(video_equipment)
		now_cascade = cv2.CascadeClassifier(xml_path)

		# 测试当前神经网络是否ok，不ok就增加训练数据restart
		while True:
			ret, frame = cap.read()
			if not ret:
				print '[END] video is end.'
				break
			frame = cv2.resize(frame, (500, 500))
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# 查找特征
			features = now_cascade.detectMultiScale(gray, 1.3, 5)
			if len(features) != 0:
				# 储存初始跟踪位置
				track_windows = []
				for (x,y,w,h) in features:
					track_windows.append((x,y,w,h))
				hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
				# 饱和度(60,180),亮度(32,255)的掩模
				mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
				# 计算颜色的直方图
				roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
				# 归一化
				cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
				while(1):
					ret ,frame = cap.read()
					frame = cv2.resize(frame, (500, 500))
					if ret == True:
						tmp_img = frame
						hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
						# 进行直方图匹配，找到位置
						dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
						# 获取所有匹配到的特点，新位置
						for i in track_windows:
							# 设置终止条件，10次迭代或至少1次移动
							ret, track_window = cv2.CamShift(dst, i, (3, 10,1))
							# 得到矩阵的四个顶点,转int0,绘制多边形
							pts = cv2.boxPoints(ret)
							pts = np.int0(pts)
							tmp_img = cv2.polylines(frame,[pts],True, 255,2)

						cv2.imshow('features',tmp_img)
						k = cv2.waitKey(25) & 0xff
						if k == ord('q'):
							break
					else:
						break

			cv2.imshow('frame',frame)
			if cv2.waitKey(25) & 0xff == ord('q'):
				break
		
		cv2.destroyAllWindows()



if __name__ == '__main__':
	reco = dynamicReco()

	# reco.trainSVMClassifier('/root/Desktop/train_car', '/root/Desktop/dog', 'car', '/root/Desktop/svm_data.dat')
	# reco.testSVMClassifier('/root/Desktop/test_data', 'car')
	# download_url = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03079136'
	# reco.videoToImg('red_car.MOV', '/root/Desktop/train_car')
	# reco.store_raw_images(download_url, '/root/Desktop/opencv_workspace/cars')	
	# reco.deleteBadImg('/root/Desktop/opencv_workspace/cars')
	# reco.createPathFile('/root/Desktop/opencv_workspace/cars')
	# reco.callOpencvSample('car_template.jpg', 'bg.txt', 'cars')
	# reco.drawFeaturesForVideo('red_car.MOV', 'data/car_cascade.xml')







