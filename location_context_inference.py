import os
import sys
import numpy as np
import argparse
import math
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import _pickle as cPickle
import librosa

import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import itertools
 

from data import *
from utils import *
from models import *
from feature_extraction import *
from enum import Enum
import csv 


import torch
torch.backends.cudnn.benchmark=True
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset

import random
# random.seed(1)

classes = ['peeling', 'chopping', 'grating', 'frying', 'filling_water', 'boiling', 'washing_dishes', 'microwave', 'garbage_disposal', 
            'blender', 'load_unload_dishes', 'typing', 'television', 'background_speech', 'vacuum', 'washing_hands', 'hair_brushing',
            'shower', 'toilet_flushing']

contexts = ['Kitchen','Living_Room','Bathroom']

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

class Context(Enum):
    Kitchen = 0
    Living_Room = 1
    Bathroom = 2


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

def get_context(label):
    if label < 11:
        context = 'Kitchen'
    elif label > 10 and label < 15:
        context = 'Living_Room'
    elif label > 14:
        context = 'Bathroom'

    return Context[context].value


def samples_per_context(data, num_samples, labels, ctxt):


    contexts_data = []
    for l in labels:
        contexts_data.append(get_context(l))

    contexts_data = np.array(contexts_data)
    if len(np.shape(data)) < 2:
        data = np.reshape(data_np, (-1,1))
    labels = np.reshape(labels,(-1,1))
    sampled_data = []
    data_np = np.hstack((data, labels))

    if len(data_np[contexts_data == ctxt]) < 1:
        return None, None

    sampled_data = data_np[contexts_data == ctxt][random.sample(list(np.arange(len(data_np[contexts_data == ctxt]))), num_samples)]

    return sampled_data[:,:-1], sampled_data[:,-1]


def infer_context(args):
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


    model_path = args.model_path
    num_samples = args.num_samples

    data, sr = load_data(args)

    results_path = './results_location_inference/data_type={}/LOPO/samples_per_context_{}/{}/'.format(data_type,num_samples,model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file = open(results_path + 'performance_results.csv', "w")
    results = csv.writer(file)
    results.writerow(["Participant", "Precision", "Recall", "F-Score", "AUC","Activities"])

    if data_type == 'recording_btw_query_answer':
        PATH = './models/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
    sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, segment_size, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)
    else:
        PATH = './models/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
    sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)
    Enc = OneHotEncoder(sparse = False)
    model_name = model

    classes_num = len(classes)
    conf = []

    all_data = []
    for t in data.participants:
        audio, label = t.data()
        all_data.extend(audio)

    for test_data in data.participants:

        print("TEST:", test_data.name)

        torch.cuda.empty_cache()

        Model = eval(model_name)

        ## CNN
        model = Model(sample_rate=sr, window_size=window_size, 
                    hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
                    classes_num=classes_num)

        # Parallel
        # print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

        checkpoint = torch.load(PATH + '/' + model_path + '_' + test_data.name + '.pth', map_location=device)

        model.load_state_dict(checkpoint)

        if 'cuda' in str(device):
            model.to(device)

        ctxt_label = []
        ctxt_pred = []
        activity_labels = []

        for i in range(10):
            for ctxt in contexts:
                print("Context {}".format(ctxt))


                data_np = []
                labels = []
                audio, label = test_data.data()
                data_np.extend(audio)
                labels.extend(label)

                if pad == 'wrap':
                    all_data_n = np.array(all_data)
                    test_data_n = np.array(data_np)

                    time_augment = max(len(row) for row in all_data_n)
                    test_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in test_data_n]

                test, test_labels = samples_per_context(test_data_padded, num_samples, labels, Context[ctxt].value)
                if test is None:
                    continue
                ctxt_label.append(Context[ctxt].value)
                activity_labels.extend([Labels(int(t)).name for t in test_labels])
                # test = np.squeeze(test)

                print("Shape of Test data: {},{}".format(np.shape(test), np.shape(test_labels)))



                test_tensor = torch.from_numpy(np.array(test)).float()
                test_tensor.to(device)
                with torch.no_grad():
                    model.eval()
                    y = model(test_tensor)

                    pred = np.argmax(np.vstack(y['clipwise_output'].cpu().numpy()), axis = 1)

                    context_pred = []
                    for p in pred:
                        context_pred.append(get_context(p))
                    majority_ctxt = Counter(context_pred).most_common(1)
                    # print(majority_ctxt[0][0])
                    ctxt_pred.append(majority_ctxt[0][0])

        # print(ctxt_label, ctxt_pred)

        Enc.fit(np.reshape([Context[c].value for c in contexts],(-1,1)))
        ctxt_label_enc = Enc.transform(np.reshape(ctxt_label,(-1,1)))
        ctxt_pred_enc = Enc.transform(np.reshape(ctxt_pred,(-1,1)))
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(ctxt_label_enc, ctxt_pred_enc, average='weighted')
        auc = None
        print('average_precision: {}'.format(precision))
        print('average recall: {}'.format(recall))
        print('average f1: {}'.format(fscore))
        print('auc: {}'.format(auc))
        results.writerow([str(test_data.name), str(precision), str(recall), str(fscore), str(auc), str(activity_labels)])

        C = confusion_matrix(ctxt_label, ctxt_pred, labels=list(Enc.categories_[0]))
        conf.append(C)
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(C, class_list=contexts, normalize=False, title='Predicted Results - {}'.format(test_data.name))

        del model
    plt.figure(figsize=(6,6))

    plot_confusion_matrix(sum(conf), class_list=contexts, normalize=True, title='Average Confusion Matrix')

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
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, rotation=90, fontsize='xx-large')
    plt.yticks(tick_marks, class_list, fontsize='xx-large')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=20,
                 color="white" if cm[i, j] > thresh else "black")
    plt.clim((0.,1.))
    plt.tight_layout()
    # plt.ylabel('Ground True Activities')
    # plt.xlabel('Predicted Activities')


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
    parser_train.add_argument('--model_path', type=str, required=True)
    parser_train.add_argument('--num_samples', type=int, required=True)

    args = parser.parse_args()

    infer_context(args)











