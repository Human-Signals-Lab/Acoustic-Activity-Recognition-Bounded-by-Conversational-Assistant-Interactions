# Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions

This is the research repository for Ok Google, What Am I Doing? Acoustic Activity Recognition Bounded by Conversational Assistant Interactions. It contains the deep learning pipeline to reproduce the results in the paper as well as the hardware and software implementations of the audio capturing device (Raspberry Pi).

## System Requirements

The deep learning system is written in `python 3`, specifically `pytorch`.

## Scripts 

### List of scripts:

- [data.py](data.py): includes the needed classes for loading the data while keeping track of participants, sessions, and activities. 
- [models.py](models.py): includes the neural networks, implemented using `pytorch`. The model used in our paper is `FineTuneCNN14`, though the script includes other models we experimented with.
- [inference.py](inference.py): loads saved models and runs inference, originally written for LOPO evaluation.
- [location_context_inference.py](location_context_inference.py): implements the location context inference analysis (Section 8.4 in the paper), i.e. inferring the location of the device from the predicted activities. 
- [voice_band_filtering.py](voice_band_filtering.py): implements voice interaction masking using REPET method (Section 8.3 in the paper) and saves the filtered data. It is an interactive script that asks to determine the lower and upper time range for where to apply the masking. 

### Running the main scripts:

## Audio Capture Device (Raspberry Pi)

### Hardware

### Software


## Reference 

To be added once paper is published.
