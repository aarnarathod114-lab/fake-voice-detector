import numpy as np
import sounddevice as sd

print("Keep talking until this finishes (about 10 seconds)...")
for i in [1, 5, 9]:
    d = sd.query_devices(i)
    try:
        rate = int(d["default_samplerate"])
        audio = sd.rec(3 * rate, samplerate=rate, channels=d["max_input_channels"],
                       device=i, blocking=True)
        print(i, d["name"], "| rate", rate, "| peak per channel:",
              np.round(np.abs(audio).max(axis=0), 4))
    except Exception as e:
        print(i, d["name"], "| ERROR:", e)