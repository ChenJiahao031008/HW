from tqdm import tqdm
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import os.path
import sys
import random
import math
import cv2

directory = '/home/chen/桌面/DeepLearning/Line-regression/dataset/label/'
dataset_train = 'train.txt'
dataset_test = 'test.txt'
images_path = []

class datasource(object):
	def __init__(self, images, poses):
		self.images = images
		self.poses = poses

def centeredCrop(img, output_side_length):
	height, width, depth = img.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	cropped_img = img[height_offset:height_offset + output_side_length,
	                          width_offset:width_offset + output_side_length]
	return cropped_img

def preprocess(images):
	images_out = [] #final result
	#Resize and crop and compute mean!
	images_cropped = []
	for i in tqdm(range(len(images))):
		X = cv2.imread(images[i])

		# X = cv2.resize(X, (224, 224))
		# X = centeredCrop(X, 224)
		images_cropped.append(X)
	#compute images mean
	N = 0
	mean = np.zeros((1, 3, 224, 224))
	for X in tqdm(images_cropped):
		mean[0][0] += X[:,:,0]
		mean[0][1] += X[:,:,1]
		mean[0][2] += X[:,:,2]
		N += 1
	mean[0] /= N
	#Subtract mean from all images
	for X in tqdm(images_cropped):
		X = np.transpose(X,(2,0,1))
		X = X - mean
		X = np.squeeze(X)
		X = np.transpose(X, (1,2,0))
		Y = np.expand_dims(X, axis=0)
		images_out.append(Y)
	return images_out


def get_data(dataset,i=0):
	poses = []
	images = []

	with open(directory+dataset) as f:
		for line in f:
			fname, p0,p1 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			poses.append((p0,p1))
			images.append(fname)
			if (i==1):
				images_path.append(fname)
	images_out = preprocess(images)
	return datasource(images_out, poses)


def getKings():
	datasource_train = get_data(dataset_train,0)
	datasource_test = get_data(dataset_test,1)


	images_train = []
	poses_train = []

	images_test = []
	poses_test = []


	for i in range(len(datasource_train.images)):
		# print(i)
		images_train.append(datasource_train.images[i])
		poses_train.append(datasource_train.poses[i])

	for i in range(len(datasource_test.images)):
		# print(i)
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_train, poses_train), datasource(images_test, poses_test)

def getKingsTest():
	datasource_train = get_data(dataset_train,0)
	datasource_test = get_data(dataset_test,1)


	images_train = []
	poses_train = []

	images_test = []
	poses_test = []

	for i in range(len(datasource_train.images)):
		# print(i)
		images_train.append(datasource_train.images[i])
		poses_train.append(datasource_train.poses[i])

	for i in range(len(datasource_test.images)):
		# print(i)
		images_test.append(datasource_test.images[i])
		poses_test.append(datasource_test.poses[i])

	return datasource(images_train, poses_train), datasource(images_test, poses_test), images_path
