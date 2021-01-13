#!/bin/bash

############# MAIN #####################################

## Comment/Uncomment target DATATYPE (name of folder where data is found) 
#DATATYPE="recording_btw_query_answer"
#DATATYPE="separated_data_interactiveV3_imputed_margin_2-10"
DATATYPE="whole_recording"
#DATATYPE="Unknown_Sounds"


#### change to folder that contains the data
FOLDER_PATH="../../Remote-User-Study/$DATATYPE/"


"""
To run inference only using saved models
Note: to run inference on unknown out-of-scope sounds, add arguments --full and --other (to use full model trained on all participant data)
"""

python3 -i inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap' #--context_location='bathroom'

