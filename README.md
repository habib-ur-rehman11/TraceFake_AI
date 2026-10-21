# TraceFake AI Base Image Authenticity Checker

AI-powered image authenticity checker that detects AI-generated and manipulated images using deep learning, EXIF analysis, and forensic techniques.

Final-year project, BS Computer Science, Virtual University of Pakistan.

## Why this project

Modern image generators produce faces and scenes that most people cannot tell apart from real photos. A detector that relies on a single cue is easy to fool, so TraceFake combines several independent signals and reports one combined prediction.

## How it works

| Signal | What it checks | Code |
|---|---|---|
| MobileNetV2 classifier | Visual patterns in the image content | `Trace_Fake_notebook.ipynb`, `models/` |
| EXIF metadata analysis | Missing or inconsistent camera metadata | `app.py` |
| Error Level Analysis (ELA) | Compression inconsistencies that suggest editing | `ela_analysis.py` |
| GAN-fingerprint cues | Frequency-level traces left by generators | `gan_fingerprint.py` |

The outputs of these signals are combined into the final prediction shown in the app (`app.py`).

## Results

- **97.7% accuracy** on the held-out test set.
- Dataset: 140K Real vs Fake Images
- Train/test split: 80/20

## Known limitation: detectors go stale

The first model worked well on its test set but failed on recent AI-generated faces downloaded from thispersondoesnotexist.com. I retrained a second model on newer images, and it classified 5 of 6 correctly.

This was a very small, informal check, not a formal evaluation, so I do not treat it as a result. What it showed me is that high test accuracy only describes performance on the generators the model was trained on. As generators improve, any detector trained on today's images will lose accuracy.

## Next steps

- Evaluate on images from generators that were not in the training data, and report results per generator instead of one overall number.
- Test which combinations of signals (ELA, GAN-fingerprint cues, classifier prediction) keep working when the generator changes.

## Setup and usage

```bash
git clone https://github.com/habib-ur-rehman11/TraceFake_AI.git
cd TraceFake_AI
pip install -r requirements.txt
```

Requirements: Python [VERSION], TensorFlow/Keras, OpenCV. [CONFIRM AND ADD A requirements.txt TO THE REPO]

## Repository structure

```
TraceFake_AI/
├── app.py                      # app: runs the signals and shows the prediction
├── ela_analysis.py             # Error Level Analysis
├── gan_fingerprint.py          # GAN-fingerprint features
├── Trace_Fake_notebook.ipynb   # training and experiments
├── models/                     # trained models
└── test.jpg                    # sample image
```

## Author

Habib Ur Rehman, BS Computer Science, Virtual University of Pakistan.
Contact: mhabibawan11@gmail.com
