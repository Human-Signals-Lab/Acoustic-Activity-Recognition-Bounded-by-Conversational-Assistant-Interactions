#!/bin/bash

############# MAIN #####################################

#DATATYPE="recording_btw_query_answer"
DATATYPE="whole_recording"
FOLDER_PATH="../../Remote-User-Study/$DATATYPE/"


## CNN Classifier
python3 main_personalized_LOPO+1_classification.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --pad='wrap' #--segment #--context_location='kitchen' #--interaction_based
