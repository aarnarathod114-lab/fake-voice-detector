from collections import Counter
from pathlib import Path
import soundfile as sf

for label in ["real", "fake"]:
    files = sorted(Path("data", label).glob("*.flac"))
    rates, channels, durations = Counter(), Counter(), []
    for f in files:
        info = sf.info(str(f))  # reads only the file header, so it's fast
        rates[info.samplerate] += 1
        channels[info.channels] += 1
        durations.append(info.duration)

    print(label.upper(), "-", len(files), "files")
    print("  sample rates:", dict(rates))
    print("  channels:", dict(channels))
    print("  duration (s): min %.1f | avg %.1f | max %.1f"
          % (min(durations), sum(durations) / len(durations), max(durations)))
    print("  example names:", [f.name for f in files[:3]])