from pathlib import Path
import librosa
import numpy as np

SR = 16000            # every clip becomes 16,000 Hz mono
SECONDS = 4           # every clip becomes exactly 4 seconds
N_SAMPLES = SR * SECONDS


def load_fixed(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        # Too short: repeat the clip instead of adding silence,
        # so the amount of silence can't give away the label
        audio = np.tile(audio, int(np.ceil(N_SAMPLES / len(audio))))
    return audio[:N_SAMPLES]


def to_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise each clip so loudness can't give away the label either
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)).astype(np.float32)


X, y, generator, source = [], [], [], []
for label_id, label in enumerate(["real", "fake"]):   # real = 0, fake = 1
    files = sorted(Path("data", label).glob("*.flac"))
    for i, f in enumerate(files, 1):
        X.append(to_mel(load_fixed(f)))
        y.append(label_id)
        parts = f.name.split("_")
        generator.append(parts[0])                  # e.g. "el" or "yt"
        source.append(parts[0] + "_" + parts[1])    # e.g. "el_0001"
        if i % 100 == 0 or i == len(files):
            print(label, i, "/", len(files), flush=True)

X = np.stack(X)
np.savez_compressed("features.npz", X=X, y=np.array(y),
                    generator=np.array(generator), source=np.array(source))
print("saved features.npz | X shape:", X.shape, "| real:", y.count(0), "| fake:", y.count(1))