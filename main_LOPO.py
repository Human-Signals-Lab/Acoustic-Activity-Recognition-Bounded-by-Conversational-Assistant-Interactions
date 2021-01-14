#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:45:09 2020

@author: rebeccaadaimi
"""

import numpy as np
import pandas as pd 
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit, LeaveOneOut
import argparse
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import matplotlib.pyplot as plt
import itertools
import csv

from sklearn import metrics
from enum import Enum
import librosa.display
import sys
from scipy import stats 
import datetime
from scipy.fftpack import dct
import _pickle as cPickle
import copy
import os

from data import *
from utils import *
from models import *
from feature_extraction import *

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

import random
random.seed(1)

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


def load_data(args):

    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model = args.model
    nb_participants = args.nb_participants
    cuda = args.cuda
    folder_path = args.folder_path
    features = args.features
    sliding_window = args.sliding_window
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    trim = args.trim
    mfcc_to_audio = args.mfcc_to_audio
    other = args.other
    experiment_two = args.experiment_two
    thresh = args.thresh
    data_type = args.data_type
    load_data = args.load_data

    participant_id = np.sort(get_sub_dirs(folder_path))

    if other or experiment_two:
        classes.append('Other')

    data_dict = {}
    data = Data()
    av_length = 0
    count = 0
    count_len = []
    for p in participant_id:
        print("Participant: {}".format(p))
        participant_folder = os.path.join(folder_path,p)
        activities = get_sub_dirs(participant_folder)
        participant = Participant(p)

        for activity in activities:
            if activity in classes:
                print("------Activity: {}".format(activity))
                activity_folder = os.path.join(participant_folder, activity)
                act = Activity(activity)

                wav_files = get_sub_filepaths(activity_folder)
                for wav in wav_files:

                    audio, sr = librosa.load(wav, sr = None, mono=True, dtype=np.float32)
                    
                    label = Labels[activity].value
                    if data_type == 'recording_btw_query_answer' and len(audio)/sr > 7.:
                        continue
                    act.add_interaction(audio, label)
                    print("Audio_length: {}".format(len(audio)/sr))
                    av_length += len(audio)
                    count_len.append(int(len(audio)/sr))
                    count = count + 1
                    data_dict[wav] = {'filename':wav, 'Activity':activity, 'label':Labels[activity].value} 
        
                participant.add_activity(act)
        if len(participant.activities) > 0:
            data.add_participant(participant)
    csv_columns = ['filename','Activity','label']
    with open('metadata.csv','w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for key, d in data_dict.items():
            writer.writerow(d)

    av_length = (av_length/count)/sr
    print('Average Length of Recordings: ', av_length)
    print('Total number of interactions: {}'.format(count))
    print('Counter Length of Audio {}'.format(Counter(count_len)))
    return data, sr


def train(args):
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model = args.model
    #checkpoint_path = args.checkpoint_path
    nb_participants = args.nb_participants
    cuda = args.cuda
    folder_path = args.folder_path
    features = args.features
    sliding_window = args.sliding_window
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    trim = args.trim
    mfcc_to_audio = args.mfcc_to_audio
    other = args.other
    experiment_two = args.experiment_two
    thresh = args.thresh
    data_type = args.data_type
    segment = args.segment
    interaction_based = args.interaction_based
    context_location = args.context_location
    pad = args.pad
    test_segment = args.test_segment
    segment_size = args.segment_size

    global classes

    if context_location == 'kitchen':
        classes = ['peeling', 'chopping', 'grating', 'frying', 'filling_water', 'boiling', 'washing_dishes', 'microwave', 'garbage_disposal', 
            'blender', 'load_unload_dishes']
    elif context_location == 'living_room':
        classes = ['typing', 'television', 'background_speech', 'vacuum']
    elif context_location == 'bathroom':
        classes = ['washing_hands', 'hair_brushing','shower', 'toilet_flushing']

    LOPO = args.LOPO

    data, sr = load_data(args)

    if model == 'Cnn6' or model =='FineTuneCNN14' or  model == 'FineTuneCnn6' or model == 'FineTuneCnn6_FC2Classifier' or model == 'FineTuneCNN14_UpdateAll' or model == 'FineTuneCNN14_BiGRU':
        model_name = model
        classes_num = len(classes)

        Enc = OneHotEncoder(sparse = False)


        if LOPO:
            loo = LeaveOneOut()

            results_path = './results_RECHECK/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/LOPO/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_batch_size={}/{}'.format(sr, window_size, hop_size, mel_bins, fmin, fmax, data_type,segment, segment_size, interaction_based, test_segment, pad, model_name, batch_size,context_location)
            if not os.path.exists(results_path):
                os.makedirs(results_path)

            file = open(results_path + 'performance_results.csv', "w")
            results = csv.writer(file)
            results.writerow(["Participant", "Precision", "Recall", "F-Score", "Accuracy"])

            for train_index, test_index in loo.split(data.participants):

                torch.cuda.empty_cache()

                Model = eval(model_name)

                ## CNN
                model = Model(sample_rate=sr, window_size=window_size, 
                            hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                            classes_num=classes_num)

                # Parallel
                print('GPU number: {}'.format(torch.cuda.device_count()))
                model = torch.nn.DataParallel(model)

                if 'cuda' in str(device):
                    model.to(device)


                test = np.asarray(data.participants)[int(test_index)]
                train = np.asarray(data.participants)[train_index.astype(int)]

                checkpoints_dir = './checkpoints/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/LOPO/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/participant_{}/batch_size={}'.format(
                            sr, window_size, hop_size, mel_bins, fmin, fmax, data_type,segment, segment_size, interaction_based, test_segment, pad, model_name,context_location,test.name,batch_size)
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
    
                statistics_path = './statistics/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/LOPO/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/participant_{}/batch_size={}/statistics.pkl'.format(
                            sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, segment_size, interaction_based, test_segment, pad, model_name,context_location,test.name, batch_size)
                if not os.path.exists(os.path.dirname(statistics_path)):
                    os.makedirs(os.path.dirname(statistics_path))

                # Statistics
                statistics_container = StatisticsContainer(statistics_path)

                print("TEST:", test.name)

                X_train = []
                y_train = []
                for t in train:
                    audio, label = t.data()
                    X_train.extend(audio)
                    y_train.extend(label)

                X_test = []
                y_test = []
                audio, label = test.data()
                X_test.extend(audio)
                y_test.extend(label)

                print(np.shape(X_train), np.shape(y_train))
                print(Counter(y_train))

                X_val, y_val = copy.deepcopy(X_test), copy.deepcopy(y_test)

                # Segment Data
                if segment == 'segment':
                    X_train, y_train = segmentation(X_train, y_train, win_length=segment_size*sr)  ### 300ms audio clips
                    X_val, y_val = segmentation(X_test, y_test, win_length=segment_size*sr)
                    if not interaction_based:
                        X_test, y_test = copy.deepcopy(X_val), copy.deepcopy(y_val)

                if segment == 'segment_half':
                    threshold = max_length = max(len(row) for row in X_train)
                    print("Max length: {}, Threshold: {}".format(threshold/sr, threshold/2/sr))
                    X_train, y_train = segmentation_half(X_train, y_train, threshold/2)  
                    X_val, y_val = segmentation_half(X_test, y_test, int(threshold/2))
                    X_test, y_test = copy.deepcopy(X_val), copy.deepcopy(y_val)


                print("Train: ", np.shape(X_train), np.shape(y_train))
                print("Train: ", Counter(y_train))

                print("Test: ", np.shape(X_test), np.shape(y_test))
                print("Test: ", Counter(y_test))

                print("Test: ", np.shape(X_val), np.shape(y_val))
                print("Test: ", Counter(y_val))


                y_train = np.reshape(y_train, (-1,1))
                Enc.fit(y_train)
                y_train = Enc.transform(y_train)

                y_test = np.reshape(y_test, (-1,1))
                y_test = Enc.transform(y_test)
                print(np.shape(y_test), np.shape(y_train))
                print(np.sum(y_test, axis = 0), np.sum(y_train, axis = 0))

                y_val = np.reshape(y_val, (-1,1))
                y_val = Enc.transform(y_val)
                print(np.shape(y_val), np.shape(y_train))
                print(np.sum(y_val, axis = 0), np.sum(y_train, axis = 0))

                print("Shape of Train Data: ", np.shape(X_train))
                print("Shape of Test Data: ", np.shape(X_test))
                print("Shape of Val Data: ", np.shape(X_val))


                if pad == 'wrap':
                    train_data = np.array(X_train)
                    test_data = np.array(X_test)

                    max_length = max(len(row) for row in train_data)
                    time_augment = max(len(row) for row in train_data)
                    time_augment = max(max(len(row) for row in test_data), time_augment)
                    print("Max Length: {}".format(max_length), time_augment)
                    train_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in train_data]

                    test_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in test_data]
                    print(np.shape(train_data_padded), np.shape(test_data_padded))

                    val_data = np.array(X_val)
                    val_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in val_data]
                    print(np.shape(train_data_padded), np.shape(val_data_padded))
                    
                elif pad == 'zero':

                    train_data = np.array(X_train)
                    max_length = max(len(row) for row in train_data)
                    train_data_padded = np.array([list(row) + [0.] * (max_length - len(row)) for row in train_data])

                    test_data= np.array(X_test)
                    max_length = max(len(row) for row in test_data)
                    test_data_padded = np.array([list(row) + [0.] * (max_length - len(row)) for row in test_data])

                    val_data= np.array(X_val)
                    max_length = max(len(row) for row in val_data)
                    val_data_padded = np.array([list(row) + [0.] * (max_length - len(row)) for row in val_data])
                elif pad == 'wrap-30':
                    train_data = np.array(X_train)
                    test_data = np.array(X_test)

                    time_augment = int(30.*sr)

                    train_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in train_data]

                    test_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in test_data]
                    print(np.shape(train_data_padded), np.shape(test_data_padded))

                    val_data = np.array(X_val)
                    val_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in val_data]
                    print(np.shape(train_data_padded), np.shape(val_data_padded))

                elif pad == 'wrap-15':
                    train_data = np.array(X_train)
                    test_data = np.array(X_test)

                    time_augment = int(15.*sr)

                    train_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in train_data]

                    test_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in test_data]
                    print(np.shape(train_data_padded), np.shape(test_data_padded))

                    val_data = np.array(X_val)
                    val_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in val_data]
                    print(np.shape(train_data_padded), np.shape(val_data_padded))                    

                x_train_tensor = torch.from_numpy(np.array(train_data_padded)).float()
                y_train_tensor = torch.from_numpy(np.array(y_train)).float()
                x_test_tensor = torch.from_numpy(np.array(test_data_padded)).float()
                y_test_tensor = torch.from_numpy(np.array(y_test)).float()

                x_val_tensor = torch.from_numpy(np.array(val_data_padded)).float()
                y_val_tensor = torch.from_numpy(np.array(y_val)).float()

                train_data = TensorDataset(x_train_tensor, y_train_tensor)
                test_data = TensorDataset(x_test_tensor, y_test_tensor)
                val_data = TensorDataset(x_val_tensor, y_val_tensor)


                train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                    batch_size=batch_size,
                                    num_workers=8, pin_memory=True, shuffle = True)

                val_loader = torch.utils.data.DataLoader(dataset=val_data, 
                                        batch_size=batch_size,
                                        num_workers=8, pin_memory=True, shuffle = True)

                if not interaction_based:
                    test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                        batch_size=batch_size,
                                        num_workers=8, pin_memory=True, shuffle = True)
                else:
                     test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                        batch_size=1,
                                        num_workers=8, pin_memory=True, shuffle = True)                   

                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

                iteration = 0
                ### Training Loop ########
                for epoch in range(num_epochs):

                    #running_loss = 0.0
                    for i, d in enumerate(train_loader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = d

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        model.train()       
                        # forward + backward + optimize
                        outputs = model(inputs)
                        clipwise_output = outputs['clipwise_output']
                        loss = F.binary_cross_entropy(clipwise_output, labels)
                        loss.backward()
                        optimizer.step()

                        if iteration % 100 == 0:   
                            print('[Epoch %d, Batch # %5d]' % (epoch + 1, i + 1))
                            print('Train loss: {}'.format(loss))
                            eval_output = []
                            true_output = []
                            test_output = []
                            true_test_output = []
                            with torch.no_grad():

                                for x_val, y_val in val_loader:

                                    x_val = torch.from_numpy(np.array(x_val)).float()
                                    x_val = x_val.to(device)
                                    y_val = y_val.to(device)

                                    model.eval()

                                    yhat = model(x_val)

                                    clipwise_output = yhat['clipwise_output']
                                    test_loss = F.binary_cross_entropy(clipwise_output, y_val)

                                    test_output.append(clipwise_output.data.cpu().numpy())
                                    true_test_output.append(y_val.data.cpu().numpy())

                                test_oo = np.argmax(np.vstack(test_output), axis = 1)
                                true_test_oo = np.argmax(np.vstack(true_test_output), axis = 1)

                                accuracy = metrics.accuracy_score(true_test_oo, test_oo)
                                precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, average='weighted')
                                try:
                                    auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="weighted")
                                except ValueError:
                                    auc_test = None
                                print('Test loss: {}'.format(test_loss))
                                print('TEST average_precision: {}'.format(precision))
                                print('TEST average f1: {}'.format(fscore))
                                print('TEST average recall: {}'.format(recall))
                                print('TEST auc: {}'.format(accuracy))

                                trainLoss = {'Trainloss': loss}
                                statistics_container.append(iteration, trainLoss, data_type='Trainloss')
                                testLoss = {'Testloss': test_loss}
                                statistics_container.append(iteration, testLoss, data_type='Testloss')
                                test_f1 = {'test_f1':fscore}
                                statistics_container.append(iteration, test_f1, data_type='test_f1')

                                statistics_container.dump()


                                checkpoint = {'iteration': iteration, 
                                                'model': model.module.state_dict(), 
                                                'optimizer': optimizer.state_dict()}

                                checkpoint_path = os.path.join(
                                                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                        
                                #torch.save(checkpoint, checkpoint_path)
                        iteration = iteration + 1
                print('Finished Training')

                ### Save model ########

                PATH = './models_RECHECK/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
                sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, segment_size, interaction_based, test_segment, pad, model_name, context_location, batch_size, learning_rate)

                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                PATH = PATH + '/{}_{}.pth'.format(model_name,test.name)
                torch.save(model.state_dict(), PATH)


                print('Inference...............')
                eval_output = []
                true_output = []
                with torch.no_grad():
                    for x_val, y_val in test_loader:

                        if test_segment:
                            x_val, _ = segmentation(x_val.data.cpu().numpy(), y_val.data.cpu().numpy(), win_length=segment_size*sr)

                        x_val = torch.from_numpy(np.array(x_val)).float()
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                    
                        model.eval()

                        yhat = model(x_val)
                        if interaction_based:
                            yhat = average_confidence(yhat, device)


                        # yhat = model(x_val)
                        clipwise_output = yhat['clipwise_output']
                        eval_output.extend(clipwise_output.data.cpu().numpy().tolist())
                        true_output.extend(y_val.data.cpu().numpy().tolist())
                        #print(clipwise_output.data.cpu().numpy())

                    eval_oo = np.argmax(np.vstack(eval_output), axis = 1)
                    true_oo = np.argmax(np.vstack(true_output), axis = 1)

                    precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_oo, eval_oo, average='weighted')
                    accuracy = metrics.accuracy_score(true_oo, eval_oo)

                    try:
                        auc = metrics.roc_auc_score(np.vstack(true_output), np.vstack(eval_output), average="weighted")
                    except ValueError:
                        auc = None
                    print('average_precision: {}'.format(precision))
                    print('average recall: {}'.format(recall))
                    print('average f1: {}'.format(fscore))
                    print('auc: {}'.format(accuracy))

                results.writerow([str(test.name), str(precision), str(recall), str(fscore), str(accuracy)])

                print(np.shape(np.argsort(true_output)[:,::-1][:,0]))
                true_y = []
                pred_y = []
                for i in range(len(np.argsort(true_output)[:,::-1][:,0])):
                    true_y.append(Enc.categories_[0][np.argsort(true_output)[[i],::-1][0][0]])
                    pred_y.append(Enc.categories_[0][np.argsort(eval_output)[[i],::-1][0][0]])

                C = confusion_matrix(true_y, pred_y)
                plt.figure(figsize=(10,10))
                plot_confusion_matrix(C, class_list=list(Enc.categories_[0]), normalize=False, title='Predicted Results - {}'.format(test.name))

                plotCNNStatistics(statistics_path, test)

                del model, optimizer, train_data, test_data, val_data, train_loader, test_loader, val_loader
            file.close()
            plt.show()






def plot_confusion_matrix(cm, class_list,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, class_list, rotation=45)
    plt.yticks(tick_marks, class_list)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Ground True Activities')
    plt.xlabel('Predicted Activities')

def plotCNNStatistics(statistics_path, test):

    statistics_dict = cPickle.load(open(statistics_path, 'rb'))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    lines = []
 
    #print(statistics_dict)
    bal_alpha = 0.3
    test_alpha = 1.0
    bal_map = np.array([statistics['Trainloss'].cpu().data.numpy() for statistics in statistics_dict['Trainloss']])    # (N, classes_num)
    test_map = np.array([statistics['Testloss'] for statistics in statistics_dict['Testloss']])    # (N, classes_num)
    test_f1 = np.array([statistics['test_f1'] for statistics in statistics_dict['test_f1']])    # (N, classes_num)

    line, = ax.plot(bal_map, color='r', alpha=bal_alpha)
    line, = ax.plot(test_map, color='r', alpha=test_alpha)

    lines.append(line)


    ax.set_ylim(0, 1.)

    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    ax.grid(color='b', linestyle='solid', linewidth=0.3)
    plt.legend(labels=['Training Loss','Testing Loss'], loc=2)
    plt.title('{}'.format(test.name))

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    line, = ax.plot(test_f1, color='r', alpha=test_alpha)
    ax.set_ylim(0,1.)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.05))
    ax.yaxis.set_ticklabels(np.around(np.arange(0, 1.01, 0.05), decimals=2))        
    plt.ylabel('Test Average Fscore')
    plt.title('{}'.format(test.name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--window_size', type=int, required=True)
    parser_train.add_argument('--hop_size', type=int, required=True)
    parser_train.add_argument('--mel_bins', type=int, required=True)
    parser_train.add_argument('--fmin', type=int, required=True)
    parser_train.add_argument('--fmax', type=int, required=True) 
    parser_train.add_argument('--model', type=str, required=True)
    #parser_inference.add_argument('--checkpoint_path', type=str, required=True)
    parser_train.add_argument('--folder_path', type=str, required=True)
    parser_train.add_argument('--features', type=str, required=True)
    parser_train.add_argument('--num_epochs', type=int, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--trim', type=int, required=True, default=0)
    parser_train.add_argument('--sliding_window', action='store_true', default=False)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--mfcc_to_audio', action='store_true', default=False)
    parser_train.add_argument('--other', action='store_true', default=False)
    parser_train.add_argument('--experiment_two', action='store_true', default=False)
    parser_train.add_argument('--thresh', type=float, required=False)
    parser_train.add_argument('--data_type', type=str, required=True)
    parser_train.add_argument('--load_data', action='store_true', default=False)
    parser_train.add_argument('--nb_participants', type=int, required=True)
    parser_train.add_argument('--LOPO', action='store_true', default=False)
    parser_train.add_argument('--segment', type=str, required=False)
    parser_train.add_argument('--interaction_based', action='store_true', default=False)
    parser_train.add_argument('--context_location', type=str, default='all')
    parser_train.add_argument('--pad', type=str, default='zero')
    parser_train.add_argument('--test_segment', action='store_true', default=False)
    parser_train.add_argument('--segment_size', type=float, required=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)




