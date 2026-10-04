"""Shared code: audio loading, spectrograms, the model, and prediction."""
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


def build_model():
    def block(c_in, c_out):
        return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
                             nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(
        block(1, 16), block(16, 32), block(32, 64), block(64, 64),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1),
    )


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


_model = None


def get_model():
    """Load the trained model once and reuse it."""
    global _model
    if _model is None:
        _model = build_model()
        _model.load_state_dict(torch.load("model.pt"))
        _model.eval()
    return _model


def predict(path):
    """Return the probability (0 to 1) that each 4-second window of the file is fake."""
    audio = load_any(path)
    if len(audio) == 0:
        raise ValueError("no audio found in this file")
    if len(audio) < N_SAMPLES:
        audio = np.tile(audio, int(np.ceil(N_SAMPLES / len(audio))))
    windows = [audio[i:i + N_SAMPLES] for i in range(0, len(audio) - N_SAMPLES + 1, N_SAMPLES)]
    batch = torch.from_numpy(np.stack([to_mel(w) for w in windows])).unsqueeze(1)
    with torch.no_grad():
        return torch.sigmoid(get_model()(batch)).squeeze(1).numpy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python detector.py <audio file>")
    p = predict(sys.argv[1])
    print("probability fake per 4-second window:", np.round(p * 100, 1))
    print("verdict:", "FAKE" if p.mean() > 0.5 else "REAL",
          "| average probability fake: %.1f%%" % (p.mean() * 100))