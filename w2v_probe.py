import random, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg, librosa, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model

SR, N_SAMPLES = 16000, 16000 * 4
LIBRISPEECH_PER_SPEAKER = 23
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SEED, HELD_OUT_GENERATOR = 0, "el"


# ---------- Part 1: prepare the audio and compute embeddings ----------
def load_fixed(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if len(audio) < N_SAMPLES:
        audio = np.tile(audio, int(np.ceil(N_SAMPLES / len(audio))))
    return audio[:N_SAMPLES]


def compress(audio, codec, bitrate):
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
    audio = audio / (np.abs(audio).max() + 1e-6) * rng.uniform(0.3, 1.5)
    audio = np.clip(audio, -1.0, 1.0)
    noise = np.random.default_rng(rng.randrange(10**9)).normal(0, rng.uniform(0, 0.005), len(audio))
    audio = (audio + noise).astype(np.float32)
    codec, bitrates = rng.choice([("libopus", ["12k", "16k", "24k", "32k"]),
                                  ("libmp3lame", ["32k", "48k", "64k"])])
    return compress(audio, codec, rng.choice(bitrates))


def process(item):
    index, path, label, gen, src = item
    audio = load_fixed(path)
    degraded = augment(audio, random.Random(index))
    return [(audio, label, gen, src, False), (degraded, label, gen, src, True)]


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

rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for n, result in enumerate(pool.map(process, items), 1):
        rows.extend(result)
        if n % 400 == 0 or n == len(items):
            print("audio prepared:", n, "/", len(items), flush=True)

w2v = Wav2Vec2Model.from_pretrained(MODEL_NAME).to("cuda").eval()
feats = []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        x = torch.from_numpy(np.stack([r[0] for r in rows[i:i + 32]])).to("cuda")
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-7)
        with torch.autocast("cuda", dtype=torch.float16):
            out = w2v(x, output_hidden_states=True)
        feats.append(torch.stack(out.hidden_states, dim=1).float().mean(dim=2).cpu().numpy())
        if (i // 32) % 40 == 0:
            print("embedded:", i, "/", len(rows), flush=True)
X = np.concatenate(feats)
y = np.array([r[1] for r in rows])
generator = np.array([r[2] for r in rows])
source = np.array([r[3] for r in rows])
augmented = np.array([r[4] for r in rows])
np.savez("w2v_features.npz", X=X.astype(np.float16), y=y, generator=generator,
         source=source, augmented=augmented)
del w2v
torch.cuda.empty_cache()


# ---------- Part 2: same split as train.py, exact EER, one probe per layer ----------
rng = np.random.default_rng(SEED)


def pick_sources(mask, fraction):
    names = np.unique(source[mask])
    rng.shuffle(names)
    return set(names[: max(1, int(len(names) * fraction))])


real = y == 0
unseen_fake = generator == HELD_OUT_GENERATOR
seen_fake = (y == 1) & ~unseen_fake
test_sources = (pick_sources(generator == "yt", 0.25) | pick_sources(generator == "ls", 0.25)
                | pick_sources(seen_fake, 0.25))
in_test = np.array([s in test_sources for s in source])
left = ~in_test
val_sources = (pick_sources((generator == "yt") & left, 0.15)
               | pick_sources((generator == "ls") & left, 0.15)
               | pick_sources(seen_fake & left, 0.15))
in_val = np.array([s in val_sources for s in source])
train = (real | seen_fake) & ~in_test & ~in_val
val = (real | seen_fake) & in_val
test_real = real & in_test
test_seen = test_real | (seen_fake & in_test)
test_unseen = test_real | unseen_fake


def eer_exact(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return (fpr[i] + fnr[i]) / 2


def fit(layer):
    scaler = StandardScaler().fit(X[train, layer])
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(scaler.transform(X[train, layer]), y[train])
    return lambda mask: clf.decision_function(scaler.transform(X[mask, layer]))


columns = [val, test_seen & ~augmented, test_seen & augmented,
           test_unseen & ~augmented, test_unseen & augmented]
print("\nEER   |   val  | seen clean | seen degr | unseen clean | unseen degr")
for layer in range(X.shape[1]):
    score = fit(layer)
    v = [eer_exact(score(m), y[m]) for m in columns]
    print(f"{layer:5d} | {v[0]:6.1%} | {v[1]:10.1%} | {v[2]:9.1%} | {v[3]:12.1%} | {v[4]:11.1%}")


# ---------- Part 3: your CNN on the same audio, with the same exact EER ----------
import detector

cnn = detector.build_model()
cnn.load_state_dict(torch.load("model.pt"))
cnn = cnn.to("cuda").eval()
mels = np.stack([detector.to_mel(r[0]) for r in rows])
with torch.no_grad():
    cnn_scores = torch.cat([cnn(torch.from_numpy(mels[i:i + 128]).unsqueeze(1).to("cuda")).squeeze(1)
                            for i in range(0, len(mels), 128)]).cpu().numpy()
v = [eer_exact(cnn_scores[m], y[m]) for m in columns]
print(f"  CNN | {v[0]:6.1%} | {v[1]:10.1%} | {v[2]:9.1%} | {v[3]:12.1%} | {v[4]:11.1%}")