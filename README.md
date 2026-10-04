# Fake Voice Detector

Detects whether a short speech clip is a real human voice or AI-generated. A small CNN
reads mel spectrograms of 4-second audio windows and outputs the probability that the
voice is synthetic. Includes a web demo where you can upload a clip and get a verdict.
**Live demo:** https://fake-voice-detector-5n3sjmtgxcsjwl6xzmarp4.streamlit.app/

## Results

| Test set | Accuracy | EER |
|---|---|---|
| Seen generators, clean audio | 98.6% | 1.9% |
| Seen generators, degraded audio | 97.7% | 1.9% |
| Unseen generator, clean audio | 91.6% | 12.9% |
| Unseen generator, degraded audio | 90.1% | 8.8% |

EER (equal error rate) is the error rate at the point where missed fakes and false alarms
are equal. Lower is better; 50% is guessing.

How the test is set up:

- **Split by source.** All clips from one speaker or recording go to exactly one of train,
  validation or test, so the model is never tested on a voice it trained on.
- **Unseen generator.** One of the six voice generators (`el`) is removed from training
  entirely and used only for testing.
- **Degraded audio.** Each clip also has a copy passed through Opus or MP3 compression with
  random loudness, clipping and noise, similar to audio sent through a phone app.

The model is good on generators it has trained on and clearly worse on one it has not.
That gap is the main open problem in this project.

## What went wrong, and how I fixed it

**1. The file format gave away the answer.** In the raw dataset every real clip was
44.1 kHz stereo and every fake clip was 16 kHz mono. I convert everything to 16 kHz mono,
4 seconds long (short clips are repeated, not padded with silence), and normalise each
spectrogram.

**2. The first model learned a shortcut.** The baseline scored 0.0% EER on seen generators
and 7.5% on the unseen one, but it called a real phone recording of my own voice 98.3%
fake. All real clips came from 14 YouTube sources, so the model had learned "sounds like
YouTube" rather than "sounds human". I added 920 clips from 40 LibriSpeech speakers and
the degraded copies described above. The same recording then scored 0.3% fake.

**3. Accuracy was near 50% while EER was under 3%.** Scores ranked clips correctly but the
decision threshold was badly off. A per-group breakdown (`diagnose.py`) showed that even
training clips were misclassified in evaluation mode, which pointed to BatchNorm behaving
differently in training and evaluation. Replacing it with GroupNorm fixed it.

**4. Longer training helped only on seen generators.** Adding a validation split,
best-epoch selection and a cosine learning-rate schedule took seen-generator EER from 3.3%
to 1.9%, but unseen-generator EER stayed about the same (12.1% to 12.9%).

The baseline numbers in point 2 used an easier test set (YouTube real clips only, no
degraded audio), so they are not directly comparable with the table above.

## Limitations

- Small dataset: 933 fake clips from 6 generators, and real speech from 14 YouTube sources
  plus 40 LibriSpeech speakers. English only.
- 9–13% EER on an unseen generator is not reliable enough for real-world use.
- Only one generator is held out, with one random seed, and the validation set has just
  78 fake clips, so small differences between runs are within noise.
- The phone-recording check is a single clip, not a benchmark.

## Data

- [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)
  (CC BY 4.0): 933 real and 933 AI-generated clips.
- [LibriSpeech dev-clean](https://www.openslr.org/12) (CC BY 4.0): read English speech,
  40 speakers.

## How to run

```
pip install torch librosa soundfile matplotlib scikit-learn huggingface_hub imageio-ffmpeg streamlit
python download_data.py          # fake/real dataset, about 1.2 GB
python download_librispeech.py   # extra real speech, 337 MB
python prepare_data.py           # spectrograms and degraded copies
python train.py                  # trains and prints the results table
     streamlit run streamlit_app.py   # web demo at http://localhost:8501
python detector.py clip.ogg      # or check one file from the terminal
```

## Next steps

- Use a pretrained speech model (wav2vec 2.0) to improve results on unseen generators.
- Train and test on more generators and a larger benchmark dataset.
- Host the demo online.