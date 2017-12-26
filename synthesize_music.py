import json
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def synthesizer(freq, duration, amp=1.0, sampling_freq=44100):
    t = np.linspace(0, duration, duration * sampling_freq)
    audio = amp * np.sin( 2* np.pi * freq * t)
    return audio.astype(np.int16)

if __name__ == '__main__':
    tone_map_file = 'freq_map.json'
    with open(tone_map_file, 'r') as f:
        tone_freq_map = json.loads(f.read())
    
    amplitude = 10000
    sampling_freq = 44100  #Hz

    tone_seq = [('D', 0.3), ('G', 0.6)]

    output = np.array([])
    for item in tone_seq:
        input_tone, duration = item
        synthesized_tone = synthesizer(tone_freq_map[input_tone], duration, amplitude, sampling_freq)
        output = np.append(output, synthesized_tone, axis=0)

    write('output_tone_seq.wav', sampling_freq, output)
