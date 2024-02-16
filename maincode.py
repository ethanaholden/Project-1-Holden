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

def audio_delay(path, d, atten):
    left = x[0]
    #print(left)
    if d != 0:
        left = np.pad(left, (d, 0), mode='constant')
        left = left[:-(d)]
    left = left*(10**(atten/10))
    print (left)
    #print (x)
    delayed3 = np.stack([left, x[1]])
    print (delayed3)
    return delayed3

if input() == 'yes':
    x, sr = librosa.load('sound.wav', mono=False)
    xside = np.rot90(x,3)
    
    outpath = r"C:\Users\joshh\AppData\Local\Programs\Python\Python311\soundout.wav"
    path = r"C:\Users\joshh\AppData\Local\Programs\Python\Python311\sound.wav"

    #delayed_audio0 = audio_delay(path, 0, 0)
    delayed_audio480 = audio_delay(path, 21, 0)
    delayed_audio1 = audio_delay(path, 44, 0)
    delayed_audio10 = audio_delay(path, 441, 0)
    delayed_audio100 = audio_delay(path, 4410, 0)
    atten0_15 = audio_delay(path, 0, -1.5)
    atten0_3 = audio_delay(path, 0, -3)
    atten0_6 = audio_delay(path, 0, -6)
    atten21_15 = audio_delay(path, 21, -1.5)
    atten21_3 = audio_delay(path, 21, -3)
    atten21_6 = audio_delay(path, 21, -6)
    

    sr = 44100
    testsr = 22050

    sf.write('teamholden-stereosoundfile-0ms.wav', xside, testsr, subtype='PCM_24')
    sf.write('teamholden-stereosoundfile-480us.wav', np.rot90(delayed_audio480,3), testsr, subtype='PCM_24')
    sf.write('teamholden-stereosoundfile-1ms.wav', np.rot90(delayed_audio1,3), testsr, subtype='PCM_24')
    sf.write('teamholden-stereosoundfile-10ms.wav', np.rot90(delayed_audio10,3), testsr, subtype='PCM_24')
    sf.write('teamholden-stereosoundfile-100ms.wav', np.rot90(delayed_audio100,3), testsr, subtype='PCM_24')

    sf.write('team[[holden]]-stereosoundfile-[[0ms]]-[[-1.5dB]].wav', np.rot90(atten0_15,3), testsr, subtype='PCM_24')
    sf.write('team[[holden]]-stereosoundfile-[[0ms]]-[[-3dB]].wav', np.rot90(atten0_3,3), testsr, subtype='PCM_24')
    sf.write('team[[holden]]-stereosoundfile-[[0ms]]-[[-6dB]].wav', np.rot90(atten0_6,3), testsr, subtype='PCM_24')
    sf.write('team[[holden]]-stereosoundfile-[[480us]]-[[-1.5dB]].wav', np.rot90(atten21_15,3), testsr, subtype='PCM_24')
    sf.write('team[[holden]]-stereosoundfile-[[480us]]-[[-3dB]].wav', np.rot90(atten21_3,3), testsr, subtype='PCM_24')
    sf.write('team[[holden]]-stereosoundfile-[[480us]]-[[-6dB]].wav', np.rot90(atten21_6,3), testsr, subtype='PCM_24')
print ("done")
