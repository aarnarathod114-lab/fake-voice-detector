import tarfile
import urllib.request
from pathlib import Path

URL = "https://openslr.trmal.net/resources/12/dev-clean.tar.gz"
archive = Path("data/dev-clean.tar.gz")
partial = Path("data/dev-clean.tar.gz.part")
target = Path("data/librispeech")

last = -5


def progress(blocks, block_size, total):
    global last
    pct = min(100, int(blocks * block_size * 100 / total))
    if pct >= last + 5:
        last = pct
        print(f"downloaded {pct}%", flush=True)


if not archive.exists():
    print("downloading 337 MB...")
    urllib.request.urlretrieve(URL, partial, progress)
    partial.rename(archive)

print("extracting...")
with tarfile.open(archive) as tar:
    tar.extractall(target, filter="data")

files = list(target.rglob("*.flac"))
speakers = {f.name.split("-")[0] for f in files}
print("done:", len(files), "clips from", len(speakers), "speakers")