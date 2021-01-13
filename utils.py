
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:45:09 2020

@author: rebeccaadaimi
"""

import os 
import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
import datetime
import _pickle as cPickle
import librosa


import numpy as np 

def create_folder(fd):
	if not os.path.exists(fd):
		os.makedirs(fd)
		
		
def get_filename(path):
	path = os.path.realpath(path)
	na_ext = path.split('/')[-1]
	na = os.path.splitext(na_ext)[0]
	return na


def get_sub_filepaths(folder):
	paths = []
	for root, dirs, files in os.walk(folder):
		for name in files:
			path = os.path.join(root, name)
			paths.append(path)
	return paths

def get_sub_folders(folder):
	paths = []
	for root, dirs, files in os.walk(folder):
		for name in dirs:
			path = os.path.join(root, name)
			paths.append(path)
	return paths

def get_sub_dirs(folder):
	return os.listdir(folder)

def concatenate(a1, a2):
	return a2 if a1 is None else np.concatenate((a1, a2))

### segment interaction audio into 300ms clips
def segmentation(data, label, win_length):
	segmented_data = []
	new_label = []

	for i in range(len(data)):
		audio = data[i]
		l = label[i]
		if len(audio) > win_length:
			segment = librosa.util.frame(audio, frame_length=int(win_length), hop_length=int(win_length), axis = 0)
			segmented_data.extend(segment)
			new_label.extend([l]*len(segment))
		else:
			segmented_data.append(audio)
			new_label.append(l)

	return segmented_data, new_label


### segment interaction audio into 300ms clips
def segmentation_half(data, label, threshold):
	segmented_data = []
	new_label = []

	for i in range(len(data)):
		audio = data[i]
		l = label[i]
		if len(audio) > threshold:
			segment = librosa.util.frame(audio, frame_length=int(len(audio)/2), hop_length=int(len(audio)/2), axis = 0)
			segmented_data.extend(segment)
			new_label.extend([l]*len(segment))
		else:
			segmented_data.append(audio)
			new_label.append(l)

	return segmented_data, new_label


def average_confidence(y, device):
	clipwise_output = y['clipwise_output'].data.cpu().numpy()
	embedding = y['embedding'].data.cpu().numpy()
	av_clipwise_output = torch.from_numpy(np.array(np.mean(clipwise_output, axis = 0).reshape(1,-1))).float()
	av_embedding = torch.from_numpy(np.array(np.mean(embedding, axis = 0))).float()

	av_clipwise_output = av_clipwise_output.to(device)
	av_embedding = av_embedding.to(device)
	output_dict = {'clipwise_output':av_clipwise_output, 'embedding':av_embedding}

	return output_dict


class Evaluator(object):
	def __init__(self, model, generator):
		self.model = model
		self.generator = generator
		
	def evaluate(self):

		# Forward
		output_dict = forward(
			model=self.model, 
			generator=self.generator, 
			return_target=True)

		clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
		target = output_dict['target']    # (audios_num, classes_num)

		average_precision = metrics.average_precision_score(
			target, clipwise_output, average=None)

		auc = metrics.roc_auc_score(target, clipwise_output, average=None)
		
		statistics = {'average_precision': average_precision, 'auc': auc}

		return statistics


class StatisticsContainer(object):
	def __init__(self, statistics_path):
		self.statistics_path = statistics_path

		self.backup_statistics_path = '{}_{}.pickle'.format(
			os.path.splitext(self.statistics_path)[0], 
			datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

		self.statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}

	def append(self, iteration, statistics, data_type):
		print(iteration)
		statistics['iteration'] = iteration
		self.statistics_dict[data_type].append(statistics)
		
	def dump(self):
		cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
		cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
		
	def load_state_dict(self, resume_iteration):
		self.statistics_dict = cPickle.load(open(self.statistics_path, 'rb'))

		resume_statistics_dict = {'Trainloss': [], 'Testloss': [], 'test_f1': []}
		
		for key in self.statistics_dict.keys():
			for statistics in self.statistics_dict[key]:
				if statistics['iteration'] <= resume_iteration:
					resume_statistics_dict[key].append(statistics)
				
		self.statistics_dict = resume_statistics_dict



