#!/bin/bash

############# MAIN #####################################

## Comment/Uncomment target DATATYPE (name of folder where data is found) 
#DATATYPE="mid-interaction_segments"
DATATYPE="whole_recording"
#DATATYPE="voice_interaction_masked"
#DATATYPE="out_of_scope_sounds"

#### change to folder that contains the data
FOLDER_PATH="../../data/$DATATYPE/"


"""
To run inference only using saved models
"""

python3 -i inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap' #--context_location='bathroom'

