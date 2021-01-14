# Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions

This is the research repository for Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions. It contains the deep learning pipeline to reproduce the results in the paper as well as the hardware and software implementations of the audio capturing device (Raspberry Pi).

## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Dataset 

The dataset is made available for download on [Texas Data Repository Dataverse](https://doi.org/10.18738/T8/OCWAZW). Use of this dataset in publications must be acknowledged by referencing our [paper](#reference).

## Pretrained Models

The Convolutional Neural Network pretrained on AudioSet that we use as a feature extractor can be downloaded [here](https://zenodo.org/record/3576403#.XveBmZM2rOQ). The one used in the paper is `Cnn14_mAP=0.431.pth`. 

## Scripts 

### List of scripts:

- [data.py](data.py): includes the needed classes for loading the data while keeping track of participants, sessions, and activities. 
- [utils.py](utils.py): includes helper functions.
- [models.py](models.py): includes the neural networks, implemented using `pytorch`. The model used in our paper is `FineTuneCNN14`, though the script includes other models we experimented with.
- [inference.py](inference.py): loads saved models and runs inference, originally written for Leave-One-Participant-Out evaluation.
- [location_context_inference.py](location_context_inference.py): implements the location context inference analysis (Section 8.4 in the paper), i.e. inferring the location of the device from the predicted activities. 
- [voice_band_filtering.py](voice_band_filtering.py): implements voice interaction masking using REPET method (Section 8.3 in the paper) and saves the filtered data. It is an interactive script that asks to determine the lower and upper time range for where to apply the masking. 
- [main_LOPO.py](main_LOPO.py), [main_LOSO.py](main_LOSO.py), [main_personalized_LOPO+1.py](main_personalized_LOPO+1.py): main scripts that run training as well as inference after training for Leave-One-Participant-Out (LOPO), Leave-One-Session-Out (LOSO), and LOPO + 1 session personalized analyses respectively. To run the scripts with required arguments, check the next [section](#running-the-main-scripts).

### Running the main scripts:

**Note that all following scripts run location-free modelling i.e. we assume the location of the device is unknown and thus train the model on all 19 classes. To switch to location-specific modelling and to speciy the location (kitchen, living room, or bathroom) add the following argument to any of the commands below `--context_location='kitchen'`**

#### Leave-One-Participant-Out
To run LOPO training and evaluation using the downloaded dataset, you can run `sudo bash runme_LOPO.sh` or more specifically, determine the data type you're using by setting DATATYPE with one of the folder names in the dataset. 

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOPO.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```
This will save LOPO models for later user as well as the inference results saved in a CSV file. The training and testing losses as well as a confusion matrix will also be plotted at the end. 
To use the mid-interaction segments, change DATATYPE to `mid-interaction_segments` to point to the corresponding data folder.

#### Personalized Analysis

This analysis relates to Section 7.3 in the paper. We run two types of analysis: (1) Leave-One-Session-Out (LOSO) analysis which essentially trains on one session and tests on the other for every participant, ultimately creating personalized models per participant, and (2) LOPO + 1 session which for every target user, the training data consisted of data from all other users in addition to data from one session of the target user, while the test data consisted of data from the other session.

To run both, you can run `sudo bash runme_personalized_analysis.sh` or more specifically:

To run LOSO:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOSO.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --pad='wrap'
``` 

To run LOPO + 1 session:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_personalized_LOPO+1.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --pad='wrap'
``` 
### Voice Interaction Masking

This analysis relates to Section 8.3 in the paper. The filtered data is included in the downloaded dataset under `voice_interaction_masked` folder. If you would like to apply the REPET voice masking on your own data, you can run the following script [voice_band_filtering.py](voice_band_filtering.py). The script is interactive in that for every audio wav file, you will be asked to input the time range (lower and upper bound) over which to apply the masking. Simply run `python3 voice_band_filtering.py`.

To run training using any of the above methods, simply set `DATATYPE="voice_interaction_masked"`.

#### Location Context Inference

This analysis relates to Section 8.4 in the paper. Using the LOPO models saved after training, you can run `sudo bash runme_location_inference.sh` or more specifically:
```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 location_context_inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap'
```

Note that the argument `--num_samples=4` determines the number of randomly selected audio clips from the hold-out participant for each context (kitchen, living room,and bathroom).

### Recognition Performance vs. Audio Length

This analysis relates to Section 8.2 in the paper. Although you can modify the original scripts above, to easily run training and evaluation using varying audio length, you can run the following commands:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOPO_VaryingAudioLength.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```

Although not included in the paper, you can run a similar analysis using LOSO:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 main_LOSO_VaryingAudioLength.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='FineTuneCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --pad='wrap'
```

### Inference

Although the main scripts include inference at the end, if you want to only run inference using the saved models, you can run `sudo bash runme_inference.sh` or more specifically:

```
DATATYPE="whole_recording"
FOLDER_PATH="../../data/$DATATYPE/"
python3 -i inference.py train --window_size=1024 --hop_size=320 --mel_bins=64 --fmin=50 --fmax=11000 --model='RaspiCNN14' --features='mfcc' --num_epochs=500 --batch_size=32 --learning_rate=1e-5 --cuda --folder_path=$FOLDER_PATH --trim=0 --data_type=$DATATYPE --nb_participants=14 --LOPO --model_path='FineTuneCNN14' --num_samples=4 --pad='wrap'
```
## Audio Capture Device (Raspberry Pi)

### Hardware

### Software


## Reference 

To be added once paper is published.
