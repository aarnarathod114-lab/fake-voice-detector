import subprocess
import sys

import imageio_ffmpeg
import librosa
import numpy as np
import torch
import torch.nn as nn

SR = 16000
SECONDS = 4
N_SAMPLES = SR * SECONDS


# Same model shape as in train.py, then load the trained weights
def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
                         nn.ReLU(), nn.MaxPool2d(2))


model = nn.Sequential(
    block(1, 16), block(16, 32), block(32, 64), block(64, 64),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1),
)
model.load_state_dict(torch.load("model.pt"))
model.eval()


def load_any(path):
    """Convert any audio file (m4a, mp3, ogg, wav, flac...) to 16 kHz mono."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def to_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)).astype(np.float32)


if len(sys.argv) < 2:
    sys.exit("usage: python predict.py <audio file>")

audio = load_any(sys.argv[1])
print("loaded %s | %.1f seconds | loudest sample: %.3f"
      % (sys.argv[1], len(audio) / SR, np.abs(audio).max()))

# Cut into 4-second windows (repeat the clip first if it's shorter than 4 seconds)
if len(audio) < N_SAMPLES:
    audio = np.tile(audio, int(np.ceil(N_SAMPLES / len(audio))))
windows = [audio[i:i + N_SAMPLES] for i in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES)]

batch = torch.from_numpy(np.stack([to_mel(w) for w in windows])).unsqueeze(1)
with torch.no_grad():
    p_fake = torch.sigmoid(model(batch)).squeeze(1).numpy()

print("probability fake per 4-second window:", np.round(p_fake * 100, 1))
mean = float(p_fake.mean())
print("verdict:", "FAKE" if mean > 0.5 else "REAL",
      "| average probability fake: %.1f%%" % (mean * 100))