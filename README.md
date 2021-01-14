# Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions

This is the research repository for Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions. It contains the deep learning pipeline to reproduce the results in the paper as well as the hardware and software implementations of the audio capturing device (Raspberry Pi).

## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Dataset 

The dataset is made available for download on [Texas Data Repository Dataverse](https://doi.org/10.18738/T8/OCWAZW). Use of this dataset in publications must be acknowledged by referencing our [paper](#reference)

## Pretrained Models

The Convolutional Neural Network pretrained on AudioSet that we use as a feature extractor can be downloaded [here](https://zenodo.org/record/3576403#.XveBmZM2rOQ). The one used in the paper is `Cnn14_mAP=0.431.pth`. 

## Scripts 

### List of scripts:

- [data.py](data.py): includes the needed classes for loading the data while keeping track of participants, sessions, and activities. 
- [models.py](models.py): includes the neural networks, implemented using `pytorch`. The model used in our paper is `FineTuneCNN14`, though the script includes other models we experimented with.
- [inference.py](inference.py): loads saved models and runs inference, originally written for Leave-One-Participant-Out evaluation.
- [location_context_inference.py](location_context_inference.py): implements the location context inference analysis (Section 8.4 in the paper), i.e. inferring the location of the device from the predicted activities. 
- [voice_band_filtering.py](voice_band_filtering.py): implements voice interaction masking using REPET method (Section 8.3 in the paper) and saves the filtered data. It is an interactive script that asks to determine the lower and upper time range for where to apply the masking. 
- [main_LOPO.py](main_LOPO.py), [main_LOSO.py](main_LOSO.py), [main_personalized_LOPO+1.py](main_personalized_LOPO+1.py): main scripts that run training as well as inference after training for Leave-One-Participant-Out (LOPO), Leave-One-Session-Out (LOSO), and LOPO + 1 session personalized analyses respectively. To run the scripts with required arguments, check the next [section](#running-the-main-scripts).

### Running the main scripts:

## Audio Capture Device (Raspberry Pi)

### Hardware

### Software


## Reference 

To be added once paper is published.
