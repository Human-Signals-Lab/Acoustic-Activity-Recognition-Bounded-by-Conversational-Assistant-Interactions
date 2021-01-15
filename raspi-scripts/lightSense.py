import picamera
import picamera.array
import numpy as np

import pyaudio
import wave
import collections
import time
import threading
import datetime
import os

            
def save_recording_v2():
    thread = threading.Thread(target=write_to_file, args=())
    thread.start()
    
def write_to_file():
    time.sleep(15)
    
    folder = datetime.datetime.now().strftime('%y-%m-%d')
    wav_output_filename = datetime.datetime.now().isoformat()[:-7] + '.wav'
    wav_output_filename = wav_output_filename.replace(":","-")
    if not os.path.exists(folder):
        os.makedirs(folder,0o777)
    path = "./" + folder + "/" + wav_output_filename
    data = FIFObuffer.copy()
    print(path)
    wavefile = wave.open(path, 'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(data))
    wavefile.close()
    print("done")

def light_sense():
    global save_to_file
    ambientLight = -1
    last_pixAverage = 1
    with picamera.PiCamera() as camera:
        print("")
        camera.resolution = (128, 80)   
        with picamera.array.PiRGBArray(camera) as stream:
            camera.exposure_mode = 'auto'
            camera.awb_mode = 'auto'
            for image in camera.capture_continuous(stream, format='rgb'):
                stream.truncate()
                stream.seek(0)
                pixAverage = int(np.average(image.array[...,1]))
                if ambientLight == -1:
                    ambientLight = pixAverage
                
                print("light meter pixAverage = %i" % pixAverage)
                
                if (pixAverage > 100) and last_pixAverage < 20:
                    save_recording_v2()
                    
                last_pixAverage = pixAverage
                


def continuous_recorder(in_data, frame_count, time_info, status):
    global FIFObuffer
    FIFObuffer.append(in_data)
    return in_data, pyaudio.paContinue

form_1 = pyaudio.paInt16
chans = 8
samp_rate = 22000
chunk = 1024
record_secs = 30
dev_index = 2
save_to_file = False
FIFObuffer = collections.deque([b"0"*2*chans*chunk]*int(samp_rate/chunk)*record_secs, maxlen=int(samp_rate/chunk)*record_secs)

audio = pyaudio.PyAudio()
stream = audio.open(format = form_1, rate = samp_rate, channels = chans, \
                        input_device_index = dev_index, input = True, \
                        frames_per_buffer = chunk, stream_callback = continuous_recorder)

light_sense()