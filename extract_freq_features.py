import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

audio_file='data/banana/banana01.wav'
sampling_freq, audio = wavfile.read(audio_file)

mfcc_features = mfcc(audio, sampling_freq, nfft=1200)
filterbank_features = logfbank(audio, sampling_freq, nfft=1200)

print('\nMFCC"\nNumber of windows=', mfcc_features.shape[0])
print('Length of each feature=', mfcc_features.shape[1])
print('\nFilter bank:\nNumboer of windows=', filterbank_features.shape[0])
print('Lengthe of each feature=', filterbank_features.shape[1])

mfcc_features = mfcc_features.T
plt.matshow(mfcc_features)
plt.title('MFCC')

    
filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')

plt.show()
