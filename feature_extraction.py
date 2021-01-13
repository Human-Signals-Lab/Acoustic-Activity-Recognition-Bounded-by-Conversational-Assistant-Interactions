#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:45:09 2020

@author: rebeccaadaimi
"""
import librosa
import numpy as np
from scipy.fftpack import dct


def librosa_logmel(args):
	# Numpy librosa
	data = args['data']
	n_fft = args['n_fft']
	hop_length = args['hop_length']
	win_length = args['win_length']
	window = args['window']
	center = args['center']
	dtype = args['dtype']
	pad_mode = args['pad_mode']

	sample_rate = args['sample_rate']
	n_mels = args['n_mels']
	fmin = args['fmin']
	fmax = args['fmax']
	ref = args['ref']
	amin = args['amin']
	top_db = args['top_db']

	np_mel_spectrogram = []
	np_logmel_spectrogram = []

	for i in range(len(data)):
		np_data = data[i,:]
		np_stft_matrix = librosa.core.stft(y=np_data, n_fft=n_fft, hop_length=hop_length, 
			win_length=win_length, window=window, center=center, dtype=dtype, 
			pad_mode=pad_mode)

		np_pad = np.pad(np_data, int(n_fft // 2), mode=pad_mode)

		np_melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
			fmin=fmin, fmax=fmax).T

		np_mel_spectrogram_i = np.dot(np.abs(np_stft_matrix.T) ** 2, np_melW)

		np_logmel_spectrogram_i = librosa.core.power_to_db(
			np_mel_spectrogram_i, ref=ref, amin=amin, top_db=top_db)

		np_mel_spectrogram.append(np_mel_spectrogram_i)
		np_logmel_spectrogram.append(np_logmel_spectrogram_i)

	return np_mel_spectrogram, np_logmel_spectrogram


def TDMFCC(args):

	_, features_train = librosa_logmel(args)
	print("Shape of Train Mel and LogMel Spectrograms: ", np.shape(features_train))

	tdmfcc = np.empty((0,args['n_mels']*4))
	for i in range(len(features_train)):
		data = features_train[i]

		stft = dct(data, n=4, axis = 0).flatten()
		tdmfcc = np.vstack((tdmfcc, stft))

	return tdmfcc



def extract_mfcc(args):
	data = args['data']
	n_fft = args['n_fft']
	hop_length = args['hop_length']
	win_length = args['win_length']
	window = args['window']
	center = args['center']
	dtype = args['dtype']
	pad_mode = args['pad_mode']

	sample_rate = args['sample_rate']
	n_mels = args['n_mels']
	fmin = args['fmin']
	fmax = args['fmax']
	ref = args['ref']
	amin = args['amin']
	top_db = args['top_db']
	n_mfcc = args['n_mfcc']
	y = args['y']

	mfcc = librosa.feature.mfcc(data, sr = sample_rate, n_mfcc = n_mfcc, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = window, center = center) #pad_mode = pad_mode,
       
	return mfcc

def extract_psc(args):
	data = args['data']
	n_fft = args['n_fft']
	hop_length = args['hop_length']
	win_length = args['win_length']
	window = args['window']
	center = args['center']
	dtype = args['dtype']
	pad_mode = args['pad_mode']

	sample_rate = args['sample_rate']
	n_mels = args['n_mels']
	fmin = args['fmin']
	fmax = args['fmax']
	ref = args['ref']
	amin = args['amin']
	top_db = args['top_db']
	n_mfcc = args['n_mfcc']
	y = args['y']
	psc_s = []
	yf = []

	psc_dict = {}
	for i in range(len(data)):
		np_data = data[i]
		psc = dct(abs(librosa.core.stft(np_data, n_fft = n_fft, hop_length = hop_length)), n=n_mfcc, axis = 0)
		psc_c.extend(psc.T)
		yf.extend([y[i]]*len(psc.T))
		psc_dict[i] = {'y':np_data, 'psc':psc, 'label':y[i]}
	return psc_c, yf, psc_dict