import wave
import sounddevice as sd
import librosa
from scipy.io.wavfile import write
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import librosa.display
import pylab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
import glob
#import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from keras import optimizers
#import keras
#from tensorflow.keras.utils import to_categorical
import time
import IPython.display as ipd
from audio2numpy import open_audio
import pydub
import soundfile as sf

duration = 5
sr = 44100
recording = sd.rec(int(duration*sr), samplerate=sr, channels=2)
print("recording...............")
sd.wait()
write("sound.wav",sr,recording)
#print(sd.query_devices())
data, sampling_rate = librosa.load("sound.wav", sr = 44100)
print(data.shape)
print(sampling_rate)
#plt.plot(data)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(data, sr=sr, color="blue")
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.show()
freq = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
print(freq.shape)
fig, ax = plt.subplots()
img = librosa.display.specshow(freq, x_axis='time', y_axis='linear', ax=ax, sr=sr, fmin=1000, fmax=8000)
ax.set(title='Spectrogram of Recording.')
fig.colorbar(img, ax=ax)
plt.show()
freq = freq[::-1]  # Invert frequency data
norm = matplotlib.colors.PowerNorm(gamma=0.5)
plt.imshow(freq, aspect='auto', cmap=plt.cm.get_cmap('inferno', 256), extent=(0, len(data) / sr, 0, 8000), norm=norm)
plt.colorbar()
plt.xlabel('Time')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram of Recording')
plt.show()

print('delay?')

def audio_delay(path):
    left = x[0]
    print(left)
    left = np.pad(left, (100, 0), mode='constant')
    left = left[:-100]
    delayed3 = np.stack([left, x[1]])
    print (delayed3)
    return delayed3

if input() == 'yes':
    x, sr = librosa.load('sound.wav', mono=False)
    xside = np.rot90(x,3)
    outpath = r"C:\Users\joshh\AppData\Local\Programs\Python\Python311\soundout.wav"
    path = r"C:\Users\joshh\AppData\Local\Programs\Python\Python311\sound.wav"
    delayed_audio10 = audio_delay(path)
    #print (delayed_audio)
    print("delayed audio shape",delayed_audio10.shape)
    print ('original shape',x.shape)
    #print (xside.shape)
    sr = 44100
    testsr = 22050
    if x.shape[1] == 2:
        print('hi')
    #savewav(delayed_audio, outpath, sr)
    sf.write('hmm.wav', xside, testsr, subtype='PCM_24')
    sf.write('hmmdel.wav', np.rot90(delayed_audio10,3), testsr, subtype='PCM_24')
    #savewav(xside, outpath, sr)
    #write('delayed_audio.wav', sr, x)
print ("done")
