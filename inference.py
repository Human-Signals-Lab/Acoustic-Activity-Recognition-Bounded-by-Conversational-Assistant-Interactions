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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    other = 19

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
        classes.append('other')

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
                    if data_type == 'mid-interaction_segments' and len(audio)/sr > 7.:
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

    global classes

    if context_location == 'kitchen':
        classes = ['peeling', 'chopping', 'grating', 'frying', 'filling_water', 'boiling', 'washing_dishes', 'microwave', 'garbage_disposal', 
            'blender', 'load_unload_dishes']
    elif context_location == 'living_room':
        classes = ['typing', 'television', 'background_speech', 'vacuum']
    elif context_location == 'bathroom':
        classes = ['washing_hands', 'hair_brushing','shower', 'toilet_flushing']

    data, sr = load_data(args)

    results_path = './results_inference_RECHECK/data_type={}/LOPO/samples_per_context_{}/{}/'.format(data_type,num_samples,model_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file = open(results_path + 'performance_results.csv', "w")
    results = csv.writer(file)
    if not other:
        results.writerow(["Participant", "Precision", "Recall", "F-Score", "AUC"])
    else:
        results.writerow(["Predicted Probability", "Predicted Label"])

    file1 = open(results_path + 'pred_prob.csv', "w")
    prob_csv = csv.writer(file1)
    prob_csv.writerow(["Predicted Probability", "Predicted Label"])



    if data_type == 'mid-interaction_segments' or data_type == 'voice_interaction_masked':
        PATH = './models/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
    sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, segment_size, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)
    else:
        if args.full and data_type=='out_of_scope_sounds':
            PATH = './models/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
     sr, window_size, hop_size, mel_bins, fmin, fmax, 'whole_recording', segment, segment_size, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)
        else:
            PATH = './models/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
            sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)
 
    # PATH = './models_RECHECK/sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}/data_type={}/segment_{}_{}_interaction_based_{}_test_segment_{}/pad_{}/{}_{}/batch_size={},lr={}'.format(
    # sr, window_size, hop_size, mel_bins, fmin, fmax, data_type, segment, segment_size, interaction_based, test_segment, pad, model_path, context_location, batch_size, learning_rate)


    Enc = OneHotEncoder(sparse = False)
    model_name = model
    print(PATH)
    #sys.exit()
    classes_num = len(classes)
    if other:
        classes_num = classes_num - 1
    conf = []

    all_data = []
    all_labels = []
    for t in data.participants:
        audio, label = t.data()
        all_data.extend(audio)
        all_labels.extend(label)


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
        if args.full:
            checkpoint = torch.load(PATH + '/' + model_path + '_FULL.pth', map_location=device)
        else:
            checkpoint = torch.load(PATH + '/' + model_path + '_' + test_data.name + '.pth', map_location=device)


        model.load_state_dict(checkpoint)

        if 'cuda' in str(device):
            model.to(device)


        data_np = []
        labels = []
        audio, label = test_data.data()
        data_np.extend(audio)
        labels.extend(label)


        labels = np.reshape(labels, (-1,1))
        Enc.fit(np.reshape(all_labels,(-1,1)))

        if other:
            Enc.fit(np.reshape(np.arange(20),(-1,1)))
        labels = Enc.transform(labels)

        if pad == 'wrap':
            all_data_n = np.array(all_data)
            test_data_n = np.array(data_np)

            time_augment = max(len(row) for row in all_data_n)
            test_data_padded = [np.pad(row, (0,time_augment-len(row)), 'wrap') for row in test_data_n]


            test_tensor = torch.from_numpy(np.array(test_data_padded)).float()
            test_tensor.to(device)
            with torch.no_grad():
                model.eval()
                y = model(test_tensor)
                prob_pred = np.vstack(y['clipwise_output'].cpu().numpy())
                pred = np.argmax(np.vstack(y['clipwise_output'].cpu().numpy()), axis = 1)

        # print(ctxt_label, ctxt_pred)

        # Enc.fit(np.reshape([Context[c].value for c in contexts],(-1,1)))
        # ctxt_label_enc = Enc.transform(np.reshape(ctxt_label,(-1,1)))
        # ctxt_pred_enc = Enc.transform(np.reshape(ctxt_pred,(-1,1)))
        precision, recall, fscore,_ = metrics.precision_recall_fscore_support(np.argmax(np.vstack(labels), axis = 1), pred, average='weighted')
        auc = None
        print('average_precision: {}'.format(precision))
        print('average recall: {}'.format(recall))
        print('average f1: {}'.format(fscore))
        print('auc: {}'.format(auc))
        if not other:
            results.writerow([str(test_data.name), str(precision), str(recall), str(fscore), str(auc)])
            for i,l in zip(prob_pred,pred):
                prob_csv.writerow([str(np.max(i)),classes[Enc.categories_[0][l]]])
        else:
            for i,l in zip(prob_pred,pred):
                results.writerow([str(np.max(i)),classes[Enc.categories_[0][l]]])


        true_y = []
        pred_y = []
        for i in range(len(np.argsort(prob_pred)[:,::-1][:,0])):
            true_y.append(Enc.categories_[0][np.argsort(labels)[[i],::-1][0][0]])
            pred_y.append(Enc.categories_[0][np.argsort(prob_pred)[[i],::-1][0][0]])

        C = confusion_matrix(true_y, pred_y, labels=list(Enc.categories_[0]))
        print("Missing Label: {}".format(np.unique(true_y)))
        # if len(C) < len(list(Enc.categories_[0])):
        #     temp = np.zeros((len(list(Enc.categories_[0])),len(list(Enc.categories_[0]))))
        #     for i in np.unique(true_y):
        #         for j in np.unique(true_y):
        #             temp[i,j] = C[i,j]
        #     conf.append(temp)
        # else:
        conf.append(C)
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(C, class_list=classes, normalize=False, title='Predicted Results - {}'.format(test_data.name))

        del model
    plt.figure(figsize=(6,6))

    plot_confusion_matrix(sum(conf), class_list=classes, normalize=False, title='Average Confusion Matrix')
    plt.tight_layout()

    plt.savefig(results_path + '/avg_confusion_matrix.png')
    file.close()
    file1.close()
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
    # plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(class_list))
    plt.xticks(tick_marks, rotation=90, fontsize='xx-large')
    plt.yticks(tick_marks, class_list, fontsize='xx-large')

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('Ground True Activities')
    # plt.xlabel('Predicted Activities')
    plt.clim((0.,1.))

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
    parser_train.add_argument('--full', action='store_true', default=False)

    args = parser.parse_args()

    infer_context(args)











