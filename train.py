import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

SEED = 0
HELD_OUT_GENERATOR = "el"   # the model never hears this generator in training
EPOCHS = 12
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

data = np.load("features.npz")
X, y, generator, source, augmented = (data[k] for k in ["X", "y", "generator", "source", "augmented"])


# ---- Split by source, so one speaker or recording never lands in both train and test ----
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
test_real = real & in_test_source
test_seen = test_real | (seen_fake & in_test_source)
test_unseen = test_real | unseen_fake

for name, mask in [("train", train), ("test_seen", test_seen), ("test_unseen", test_unseen)]:
    print(f"{name}: {int((mask & real).sum())} real, {int((mask & ~real).sum())} fake")


# ---- Model: a small CNN that reads the spectrogram like an image ----
def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out),
                         nn.ReLU(), nn.MaxPool2d(2))


model = nn.Sequential(
    block(1, 16), block(16, 32), block(32, 64), block(64, 64),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1),
)


def tensors(mask):
    return torch.from_numpy(X[mask]).unsqueeze(1), torch.from_numpy(y[mask]).float()


# ---- Train ----
Xtr, ytr = tensors(train)
# There are more real clips than fake ones, so mistakes on fakes count for more
pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(order), 32):
        idx = order[i:i + 32]
        optimizer.zero_grad()
        loss = loss_fn(model(Xtr[idx]).squeeze(1), ytr[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch:2d} | train loss {total / len(Xtr):.3f}", flush=True)


# ---- Test ----
def evaluate(name, mask):
    Xt, yt = tensors(mask)
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(Xt[i:i + 64]).squeeze(1) for i in range(0, len(Xt), 64)]).numpy()
    labels = yt.numpy()
    accuracy = ((scores > 0) == (labels == 1)).mean()
    fpr, tpr, _ = roc_curve(labels, scores)
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    print(f"{name}: accuracy {accuracy:.1%} | EER {eer:.1%}")


evaluate("seen generators, clean    ", test_seen & ~augmented)
evaluate("seen generators, degraded ", test_seen & augmented)
evaluate("unseen generator, clean   ", test_unseen & ~augmented)
evaluate("unseen generator, degraded", test_unseen & augmented)

torch.save(model.state_dict(), "model.pt")
print("saved model.pt")