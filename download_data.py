import posixpath
from collections import Counter
from huggingface_hub import HfApi, snapshot_download

REPO = "garystafford/deepfake-audio-detection"

# Look at what's in the dataset before downloading
info = HfApi().dataset_info(REPO, files_metadata=True)
files = info.siblings
total_mb = sum((f.size or 0) for f in files) / 1e6
print("files:", len(files), "| total size: %.0f MB" % total_mb)

folders = Counter(
    posixpath.dirname(f.rfilename) + "  (." + f.rfilename.rsplit(".", 1)[-1] + ")"
    for f in files
)
for name, count in folders.items():
    print("  ", name, "->", count, "files")

# Download everything into a local "data" folder
path = snapshot_download(REPO, repo_type="dataset", local_dir="data")
print("downloaded to:", path)