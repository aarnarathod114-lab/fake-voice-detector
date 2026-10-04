import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np

SR = 16000
SECONDS = 4
N_SAMPLES = SR * SECONDS
LIBRISPEECH_PER_SPEAKER = 23   # 40 speakers x 23 = 920 clips, about as many as YouTube
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_fixed(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.tile(audio, int(np.ceil(N_SAMPLES / len(audio))))
    return audio[:N_SAMPLES]


def to_mel(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)).astype(np.float32)


def compress(audio, codec, bitrate):
    """Squeeze audio through a lossy codec and back, like a phone app does."""
    container = {"libopus": "ogg", "libmp3lame": "mp3"}[codec]
    encoded = subprocess.run(
        [FFMPEG, "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-c:a", codec, "-b:a", bitrate, "-f", container, "-"],
        input=audio.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    decoded = subprocess.run(
        [FFMPEG, "-v", "error", "-i", "-", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-"],
        input=encoded, capture_output=True, check=True).stdout
    out = np.frombuffer(decoded, dtype=np.float32)
    if len(out) < N_SAMPLES:
        out = np.tile(out, int(np.ceil(N_SAMPLES / len(out))))
    return out[:N_SAMPLES]


def augment(audio, rng):
    """Make a degraded copy: random loudness (may clip), a little noise, then compression."""
    audio = audio / (np.abs(audio).max() + 1e-6) * rng.uniform(0.3, 1.5)
    audio = np.clip(audio, -1.0, 1.0)
    noise = np.random.default_rng(rng.randrange(10**9)).normal(0, rng.uniform(0, 0.005), len(audio))
    audio = (audio + noise).astype(np.float32)
    codec, bitrates = rng.choice([("libopus", ["12k", "16k", "24k", "32k"]),
                                  ("libmp3lame", ["32k", "48k", "64k"])])
    return compress(audio, codec, rng.choice(bitrates))


def process(item):
    index, path, label, gen, src = item
    rng = random.Random(index)
    audio = load_fixed(path)
    rows = [(to_mel(audio), label, gen, src, False)]
    try:
        rows.append((to_mel(augment(audio, rng)), label, gen, src, True))
    except subprocess.CalledProcessError:
        pass   # counted below as a missing degraded copy
    return rows


# ---- Build the list of clips: (path, label, generator, source) ----
clips = []
for f in sorted(Path("data/real").glob("*.flac")):
    p = f.name.split("_")
    clips.append((f, 0, "yt", p[0] + "_" + p[1]))

by_speaker = {}
for f in sorted(Path("data/librispeech").rglob("*.flac")):
    by_speaker.setdefault(f.name.split("-")[0], []).append(f)
for speaker, files in by_speaker.items():
    for f in files[:LIBRISPEECH_PER_SPEAKER]:
        clips.append((f, 0, "ls", "ls_" + speaker))

for f in sorted(Path("data/fake").glob("*.flac")):
    p = f.name.split("_")
    clips.append((f, 1, p[0], p[0] + "_" + p[1]))

items = [(i, *c) for i, c in enumerate(clips)]
print("clips to process:", len(items))

# ---- Process them, 4 at a time ----
rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for n, result in enumerate(pool.map(process, items), 1):
        rows.extend(result)
        if n % 200 == 0 or n == len(items):
            print(n, "/", len(items), flush=True)

X = np.stack([r[0] for r in rows])
y = np.array([r[1] for r in rows])
augmented = np.array([r[4] for r in rows])
np.savez_compressed("features.npz", X=X, y=y,
                    generator=np.array([r[2] for r in rows]),
                    source=np.array([r[3] for r in rows]),
                    augmented=augmented)
print("saved features.npz | X shape:", X.shape)
print("real:", int((y == 0).sum()), "| fake:", int((y == 1).sum()),
      "| degraded copies:", int(augmented.sum()), "of", len(items))