import soundfile as sf
import numpy as np

sr = 16000
duration = 3
t = np.linspace(0, duration, int(sr*duration))
audio = 0.1*np.sin(2*np.pi*220*t)

sf.write("test.wav", audio, sr)
