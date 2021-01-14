#!/bin/bash

############# MAIN #####################################
#DATATYPE="mid-interaction_segments"
DATATYPE="whole_recording"
#DATATYPE="voice_interaction_masked"
FOLDER_PATH="../../data/$DATATYPE/"

## CNN Classifier
python3 main_LOPO.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'



"""
To run LOPO with varying audio length: replace main_LOPO.py with main_LOPO_VaryingAudioLength.py
"""
#python3 main_LOPO_VaryingAudioLength.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'

