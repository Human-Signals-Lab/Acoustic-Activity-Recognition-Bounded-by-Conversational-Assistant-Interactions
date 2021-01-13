#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:45:09 2020

@author: rebeccaadaimi
"""

import numpy as np 
from utils import *

class Data:

    def __init__(self):
        self.participants = []

    def num_people(self):
        return len(self.participants)

    def add_participant(self, participant):
        self.participants.append(participant)

    def data(self, excluded_participants=None):
        if excluded_participants is None:
            excluded_participants = []
        audio, labels = [],[]
        for participant in [participant for participant in self.participants if
                            participant not in excluded_participants]:
            p_audio, p_labels = participant.data()
            audio.append(p_audio)
            labels.append(p_labels)
        return audio, labels

    def dataset(self, excluded_participants=None):
        audio, labels = self.data(excluded_participants)
        return Dataset(audio, labels)



class Participant:

    def __init__(self, name):
        self.name = name
        self.activities = []
        self.session1 = []
        self.session2 = []

    def add_activity(self, activity):
        self.activities.append(activity)

    def data(self):
        audio, labels = [], []

        for i in range(len(self.activities)):
            a_audio, a_labels = self.activities[i].data()
            audio.extend(a_audio)
            labels.extend(a_labels)

        return audio, labels

    def session_data(self):
        self.session1 = Session()
        self.session2 = Session()
        for i in range(len(self.activities)):
            a_audio, a_labels = self.activities[i].data()
            if len(a_labels) < 1:
                continue
            self.session1.add_activity(a_audio[:-1], a_labels[:-1])
            if len(a_labels) > 1:
                self.session2.add_activity(a_audio[-1:], a_labels[-1:])

        return self.session1, self.session2         

class Session:
    def __init__(self):
        self.activities = []

    def add_activity(self, audio, label):
        if len(label) > 1:
            for i in range(len(audio)):
                self.activities.append(Interaction(audio[i], label[i]))
        elif len(label) == 1:
           self.activities.append(Interaction(audio[0], label[0]))


    def data(self):
        audio, labels = [], []
        for i in range(len(self.activities)):
            i_audio, i_labels = self.activities[i].audio, self.activities[i].label
            audio.append(i_audio)
            labels.append(i_labels)

        return audio, labels


class Activity:
    def __init__(self, activity):
        self.activity = activity
        self.interactions = []

    def add_interaction(self, audio, label):
        self.interactions.append(Interaction(audio, label))

    def data(self):
        audio, labels = [], []
        for i in range(len(self.interactions)):
            i_audio, i_labels = self.interactions[i].audio, self.interactions[i].label
            audio.append(i_audio)
            labels.append(i_labels)

        return audio, labels

class Interaction:
    def __init__(self, audio, label):
        self.audio = audio
        self.label = label




