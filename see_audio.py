import matplotlib
matplotlib.use("Agg")  # save images to a file instead of opening a window

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load a sample speech clip that comes with librosa (downloads once)
audio, sr = librosa.load(librosa.ex("libri1"), sr=16000, duration=4)
print("samples:", audio.shape, "| sample rate:", sr, "| seconds:", len(audio) / sr)

# Turn the audio into a mel spectrogram (time on x-axis, pitch on y-axis)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
mel_db = librosa.power_to_db(mel, ref=np.max)
print("spectrogram shape:", mel_db.shape)

# Save it as an image
plt.figure(figsize=(8, 3))
librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel spectrogram")
plt.tight_layout()
plt.savefig("spectrogram.png")
print("saved spectrogram.png")