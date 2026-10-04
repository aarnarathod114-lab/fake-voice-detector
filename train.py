import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve

SEED = 0
HELD_OUT_GENERATOR = "el"   # the model never hears this generator in training
EPOCHS = 24
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

data = np.load("features.npz")
X, y, generator, source, augmented = (data[k] for k in ["X", "y", "generator", "source", "augmented"])


# ---- Split by source, so one speaker or recording never lands in two splits ----
def pick_sources(mask, fraction):
    names = np.unique(source[mask])
    rng.shuffle(names)
    return set(names[: max(1, int(len(names) * fraction))])


real = y == 0
unseen_fake = generator == HELD_OUT_GENERATOR
seen_fake = (y == 1) & ~unseen_fake

# Test split first (same as before, so results stay comparable)
test_sources = (pick_sources(generator == "yt", 0.25) | pick_sources(generator == "ls", 0.25)
                | pick_sources(seen_fake, 0.25))
in_test = np.array([s in test_sources for s in source])

# Then a validation split, taken from what's left
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

for name, mask in [("train", train), ("val", val), ("test_seen", test_seen), ("test_unseen", test_unseen)]:
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


def scores_for(mask):
    Xt, yt = tensors(mask)
    model.eval()
    with torch.no_grad():
        s = torch.cat([model(Xt[i:i + 64]).squeeze(1) for i in range(0, len(Xt), 64)]).numpy()
    return s, yt.numpy()


def eer_of(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]


# ---- Train, checking the validation set after every epoch ----
Xtr, ytr = tensors(train)
pos_weight = (ytr == 0).sum() / (ytr == 1).sum()   # more real than fake clips, so weight fakes up
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_eer, best_epoch, best_state = 1.0, 0, None
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
    scheduler.step()

    val_eer = eer_of(*scores_for(val))
    if val_eer <= best_eer:
        best_eer, best_epoch, best_state = val_eer, epoch, copy.deepcopy(model.state_dict())
    print(f"epoch {epoch:2d} | train loss {total / len(Xtr):.3f} | val EER {val_eer:.1%}", flush=True)

model.load_state_dict(best_state)
print(f"keeping the model from epoch {best_epoch} (val EER {best_eer:.1%})")


# ---- Test ----
def evaluate(name, mask):
    scores, labels = scores_for(mask)
    accuracy = ((scores > 0) == (labels == 1)).mean()
    print(f"{name}: accuracy {accuracy:.1%} | EER {eer_of(scores, labels):.1%}")


evaluate("seen generators, clean    ", test_seen & ~augmented)
evaluate("seen generators, degraded ", test_seen & augmented)
evaluate("unseen generator, clean   ", test_unseen & ~augmented)
evaluate("unseen generator, degraded", test_unseen & augmented)

torch.save(model.state_dict(), "model.pt")
print("saved model.pt")