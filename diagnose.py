import numpy as np
import torch
import torch.nn as nn

SEED = 0
HELD_OUT_GENERATOR = "el"
rng = np.random.default_rng(SEED)

data = np.load("features.npz")
X, y, generator, source, augmented = (data[k] for k in ["X", "y", "generator", "source", "augmented"])


# Recreate exactly the same train/test split as train.py
def pick_test_sources(mask, fraction=0.25):
    names = np.unique(source[mask])
    rng.shuffle(names)
    return set(names[: max(1, int(len(names) * fraction))])


real = y == 0
unseen_fake = generator == HELD_OUT_GENERATOR
seen_fake = (y == 1) & ~unseen_fake
test_sources = (pick_test_sources(generator == "yt") | pick_test_sources(generator == "ls")
                | pick_test_sources(seen_fake))
in_test_source = np.array([s in test_sources for s in source])
train = (real | seen_fake) & ~in_test_source
test = ((real | seen_fake) & in_test_source) | unseen_fake


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
                         nn.ReLU(), nn.MaxPool2d(2))


model = nn.Sequential(
    block(1, 16), block(16, 32), block(32, 64), block(64, 64),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1),
)
model.load_state_dict(torch.load("model.pt"))
model.eval()

with torch.no_grad():
    scores = torch.cat([model(torch.from_numpy(X[i:i + 128]).unsqueeze(1)).squeeze(1)
                        for i in range(0, len(X), 128)]).numpy()

groups = [("YouTube real", generator == "yt"), ("LibriSpeech real", generator == "ls"),
          ("seen-generator fake", seen_fake), ("unseen-generator fake", unseen_fake)]

print(f"{'split':6s} {'group':22s} {'quality':9s} {'clips':>5s} {'called fake':>12s} {'median score':>13s}")
for split_name, split in [("train", train), ("test", test)]:
    for group_name, group in groups:
        for quality, q in [("clean", ~augmented), ("degraded", augmented)]:
            m = split & group & q
            if m.sum() == 0:
                continue
            print(f"{split_name:6s} {group_name:22s} {quality:9s} {int(m.sum()):5d} "
                  f"{(scores[m] > 0).mean():11.1%} {np.median(scores[m]):13.2f}")