import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

audio_file='data/apple/apple01.wav'
sampling_freq, audio = wavfile.read(audio_file)

print('\nShape:', audio.shape)
print('Sampling frequency', sampling_freq)
print('Datatype:', audio.dtype)
print('Duration:', round(audio.shape[0] / float(sampling_freq), 3), 'seconds')

audio = audio / (2. ** 15)

print("Len audio", len(audio))
audio = audio[:240]

x_values = np.arange(0, len(audio), 1) / float(sampling_freq)

x_values *= 1000

plt.plot(x_values, audio, color='black')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Audio signal')
plt.show()
