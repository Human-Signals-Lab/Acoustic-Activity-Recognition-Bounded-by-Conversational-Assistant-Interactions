## Voice Band Stripping: Band Pass Filter; F_low = 300Hz, F_high = 3KHz 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:45:09 2020

@author: rebeccaadaimi
"""

import numpy as np 
import pandas as pd 
from scipy.io import wavfile
from enum import Enum
import random
random.seed(1)
from utils import *
from scipy.signal import butter, lfilter, filtfilt
import os
import scipy
from librosa import display
import matplotlib.pyplot as plt
from collections import Counter
#from foreground_background_separation import foreground_background_separation
classes = ['peeling', 'chopping', 'grating', 'frying', 'filling_water', 'boiling', 'washing_dishes', 'microwave', 'garbage_disposal', 
            'blender', 'load_unload_dishes', 'typing', 'television', 'background_speech', 'vacuum', 'washing_hands', 'hair_brushing',
            'shower', 'toilet_flushing']

class Labels(Enum):
    peeling = 0
    chopping = 1
    grating = 2
    frying = 3
    filling_water = 4
    boiling = 5
    washing_dishes = 6
    microwave = 7
    garbage_disposal = 8
    blender = 9
    load_unload_dishes = 10
    typing = 11
    television = 12
    background_speech = 13
    vacuum = 14
    washing_hands = 15
    hair_brushing = 16
    shower = 17
    toilet_flushing = 18


class foreground_background_separation():
    
    def __init__(self,
                 # path: str,
                 margin_i: int,
                 margin_v: int,
                 power: int):
        
        # self.path = path
        self.margin_i = margin_i
        self.margin_v = margin_v
        self.power = power
        
    
    def separate(self, filename, lower_bound, upper_bound): 
        self.filename = filename     
        # sr = 32000
        self.y, self.sr = librosa.load(filename, sr=None, mono= True)
        self.getMagPhase(lower_bound,upper_bound)
        
        self.filter()

    def impute(self, filename, lower_bound, upper_bound): 
        self.filename = filename     
        # sr = 32000
        self.y, self.sr = librosa.load(filename, sr=None, mono= True)
        self.getMagPhase(lower_bound,upper_bound)
        
        self.impute_data()
            
    def getMagPhase(self,lower_bound,upper_bound):
        self.S_full, self.phase = librosa.magphase(librosa.stft(self.y))
        self.idx = slice(*librosa.time_to_frames([int(lower_bound),int(upper_bound)], sr=self.sr))
        
        
    def filter(self):
        self.S_filter_full = librosa.decompose.nn_filter(self.S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=self.sr)))

        #print(np.shape(self.S_filter))
        self.S_filter = np.minimum(self.S_full[:,self.idx], self.S_filter_full[:,self.idx])
        

        self.mask_i = librosa.util.softmask(self.S_filter,
                                self.margin_i * (self.S_full[:,self.idx] - self.S_filter),
                                power = self.power)

        self.mask_v = librosa.util.softmask(self.S_full[:,self.idx] - self.S_filter,
                               self.margin_v * self.S_filter,
                               power = self.power)

        self.mask_full_i = np.ones(np.shape(self.S_full))
        self.mask_full_v = np.zeros(np.shape(self.S_full))
        self.mask_full_i[:,self.idx] = self.mask_i
        self.mask_full_v[:,self.idx] = self.mask_v

        self.S_foreground = self.mask_full_v * self.S_full
        self.S_background = self.mask_full_i * self.S_full

        self.impute_values()

        D_foreground = self.S_foreground * self.phase
        D_background = self.S_background * self.phase

        D_background_imputed = self.S_background_imputed * self.phase
        
        self.y_background_imputed = librosa.istft(D_background_imputed)
        self.y_foreground = librosa.istft(D_foreground)
        self.y_background = librosa.istft(D_background)
     
    def get_S_Foreground(self):
        return self.S_foreground
    
    def get_S_Background(self):
        return self.S_background
    
    def get_S_Full(self):
        return self.S_full
    
    def plotSpectrums(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_full, ref=np.max),
                                 y_axis='log', sr=self.sr)
        plt.title('Full spectrum')
        plt.colorbar()
        
        plt.subplot(4, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_background, ref=np.max),
                                 y_axis='log', sr=self.sr)
        plt.title('Background')
        plt.colorbar()
        plt.subplot(4, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(self.S_foreground, ref=np.max),
                                 y_axis='log', sr=self.sr)
        plt.title('Foreground')
        plt.colorbar()

        plt.subplot(4, 1, 4)

        librosa.display.specshow(librosa.amplitude_to_db(self.S_background_imputed, ref=np.max),
                                 y_axis='log',x_axis='time', sr=self.sr)
        plt.colorbar()
        plt.tight_layout()
        #plt.show()
 
    def plotLogMelSpectrograms(self):
        plt.figure(figsize=(12, 8))
        S = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels = 64)
        S_db = librosa.power_to_db(S, ref=np.max)
        S = librosa.feature.melspectrogram(y=self.y_background, sr=self.sr, n_mels = 64)
        S_back = librosa.power_to_db(S, ref=np.max)    
        S = librosa.feature.melspectrogram(y=self.y_foreground, sr=self.sr, n_mels = 64)
        S_fore = librosa.power_to_db(S, ref=np.max)
        S = librosa.feature.melspectrogram(y = self.y_background_imputed, sr= self.sr, n_mels=64)
        S_back_imputed = librosa.power_to_db(S,ref=np.max)
        print(min(S_db.reshape((-1,1))),min(S_back.reshape((-1,1))),min(S_fore.reshape((-1,1))),min(S_back_imputed.reshape((-1,1))))
        min_c = min([min(S_db.reshape((-1,1))),min(S_back.reshape((-1,1))),min(S_fore.reshape((-1,1))),min(S_back_imputed.reshape((-1,1)))])
        plt.subplot(4, 1, 1)

        librosa.display.specshow(S_db, x_axis='time', y_axis='mel',sr=self.sr,vmin=min_c)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Original Log-mel Spectrogram')
        
        plt.subplot(4, 1, 2)

        librosa.display.specshow(S_back, x_axis='time', y_axis='mel',sr=self.sr,vmin=min_c)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Background Log-mel Spectrogram')
        plt.subplot(4, 1, 3)


        librosa.display.specshow(S_back_imputed, x_axis='time', y_axis='mel',sr=self.sr,vmin=min_c)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Background Log-mel Spectrogram (after audio imputation)')

        plt.subplot(4, 1, 4)

        librosa.display.specshow(S_fore, x_axis='time', y_axis='mel',sr=self.sr,vmin=min_c)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Foreground Log-mel Spectrogram')
        plt.tight_layout()

        plt.show()

    def plotMedianModel(self):
        plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(self.S_filter_back, ref=np.max),
                                 y_axis='log',x_axis='time', sr=self.sr)
        plt.colorbar()
        plt.tight_layout()
        #plt.show()

    def plotMasks(self):
        plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(self.mask_full_i, ref=np.max),
                                 y_axis='log',x_axis='time', sr=self.sr)
        plt.colorbar()
        plt.figure()
        librosa.display.specshow(librosa.amplitude_to_db(self.mask_full_v, ref=np.max),
                                 y_axis='log',x_axis='time', sr=self.sr)
        plt.colorbar()
        plt.tight_layout()
        plt.show()        

    def writeWav(self):
        filename = self.filename.split('/')[-1]
        librosa.output.write_wav('./background-foreground-extraction/' + filename[:-4] + '_extracted_foreground_audio_margin_' + str(self.margin_v) + '.wav',self.y_foreground,self.sr)
        librosa.output.write_wav('./background-foreground-extraction/' + filename[:-4] + '_extracted_background_audio_margin_' + str(self.margin_i) + '.wav',self.y_background,self.sr)

    def impute_values(self):
        self.S_filter_back = librosa.decompose.nn_filter(self.S_background,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(2, sr=self.sr)))

        mask = (self.mask_full_i < 0.5) & (self.mask_full_v > .1)
        self.S_background_imputed = self.S_background.copy()
        self.S_background_imputed[mask] = self.S_filter_back[mask]
  
def filter_save_data(folder_path):

    participant_id = np.sort(get_sub_dirs(folder_path))

    filtered_path = folder_path.split('/')[:-2]
    filtered_path.append('filtered_data_bandstop_{}-{}'.format(str(80),str(4000)))
    filtered_path = '/'.join(filtered_path)
    for p in participant_id:
        print("Participant: {}".format(p))
        participant_folder = os.path.join(folder_path,p)
        filtered_path1 = os.path.join(filtered_path,p)
        activities = get_sub_dirs(participant_folder)

        for activity in activities:
            if activity in classes:
                print("------Activity: {}".format(activity))
                activity_folder = os.path.join(participant_folder, activity)
                filtered_path2 = os.path.join(filtered_path1,activity)

                wav_files = get_sub_filepaths(activity_folder)
                for wav in wav_files:
                    filtered_path3 = os.path.join(filtered_path2,wav.split('/')[-1])

                    audio, sr = librosa.load(wav, sr = None, mono=True, dtype=np.float32)
                

                    filtered_data = butter_bandpass_filter(audio, 80, 4000, sr)

                    filtered_path3 = filtered_path3.split('/')
                    filt_path = filtered_path3[:-1]
                    filename=  filtered_path3[-1]
                    filt_path = '/'.join(filt_path)
                    
                    save_filtered_data(filtered_data, filt_path, filename,sr)


def fore_back_sep_data(folder_path):

    participant_id = np.sort(get_sub_dirs(folder_path))


    ## if have lower and upper bounds saved in excel file
    #bounds_path = 'foreground_background_voice_separation_bounds.xlsx'
    #bounds = pd.read_excel(bounds_path)

    margin_i, margin_v = 2,10
    power = 2
    filtered_path = folder_path.split('/')[:-2]
    filtered_path.append('separated_data_interactiveV3_imputed_margin_{}-{}'.format(margin_i,margin_v))
    filtered_path = '/'.join(filtered_path)
    separator = foreground_background_separation(margin_i,margin_v,power)
    count = 0
    for p in participant_id:

        print("Participant: {}".format(p))
        participant_folder = os.path.join(folder_path,p)
        filtered_path1 = os.path.join(filtered_path,p)
        activities = get_sub_dirs(participant_folder)

        for activity in activities:
            if activity in classes:

                print("------Activity: {}".format(activity))
                activity_folder = os.path.join(participant_folder, activity)
                filtered_path2 = os.path.join(filtered_path1,activity)

                wav_files = get_sub_filepaths(activity_folder)
                for wav in wav_files:
                    filtered_path3 = os.path.join(filtered_path2,wav.split('/')[-1])

                    audio, sr = librosa.load(wav, sr = None, mono=True, dtype=np.float32)

                    filtered_path3 = filtered_path3.split('/')
                    filt_path = filtered_path3[:-1]
                    filename=  filtered_path3[-1]
                    filt_path = '/'.join(filt_path)
                    print(filt_path, filename)

                    # lower_bound = input("Please enter lower bound: ")
                    # upper_bound = input("Please enter upper bound: ")
                    #lower_bound, upper_bound = bounds['lower bound'].iloc[count], bounds['upper bound'].iloc[count]   ### if lower and upper bounds loaded from excel file (comment out the two input lines right before)
                    print(lower_bound, upper_bound)
                    count = count + 1
                    separator.separate(wav, lower_bound, upper_bound) 
          
                    separator.plotLogMelSpectrograms()

                    save_filtered_data(separator.y_background_imputed, filt_path, filename,separator.sr)

def load_data(folder_path):

    participant_id = np.sort(get_sub_dirs(folder_path))
    av_length = 0
    count = 0
    count_len = []
    for p in participant_id:
        print("Participant: {}".format(p))
        participant_folder = os.path.join(folder_path,p)
        activities = get_sub_dirs(participant_folder)

        for activity in activities:
            if activity in classes:

                print("------Activity: {}".format(activity))
                activity_folder = os.path.join(participant_folder, activity)

                wav_files = get_sub_filepaths(activity_folder)
                for wav in wav_files:

                    audio, sr = librosa.load(wav, sr = None, mono=True, dtype=np.float32)
                    if len(audio)/sr > 7.:
                        continue
                    av_length += len(audio)
                    count_len.append(len(audio)/sr)
                    count = count + 1
    av_length = (av_length/count)/sr
    avg_length = np.mean(count_len)
    std_length = np.std(count_len)
    print('Average Length of Recordings: ', av_length)
    print('Total number of interactions: {}'.format(count))
    print('Mean {} and std {}'.format(avg_length,std_length))
    
def fft_plot(data, sr, activity):
    n = len(data)
    T = 1./sr
    yf = scipy.fft.fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), int(n/2))
    fig, ax = plt.subplots()
    ax.plot(xf[:], 2.0/n*np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title(activity)
    return 

def save_filtered_data(data,path,filename,sr):
    if not os.path.exists(path):
        os.makedirs(path)
    normalized_data = data / np.abs(data).max()
    wavfile.write(os.path.join(path,filename), sr, normalized_data.astype(np.float32))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=10):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':

    folder_path = '../../Remote-User-Study/whole_recording/'
    #filter_save_data(folder_path)
    fore_back_sep_data(folder_path)
    #load_data('../../Remote-User-Study/recording_btw_query_answer/')





