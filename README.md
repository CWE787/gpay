# =========================
# README (quick start)
# =========================
"""
Quick Start
-----------
1) Create env & install deps
   python -m venv venv
   source venv/bin/activate  (Windows: venv\Scripts\activate)
   pip install -r requirements.txt

2) Register a user (3s capture, keeps top-12 quality frames)
   python register.py --name "Sai" --phone 9876543210 --device 0

   If you get a quality warning, improve lighting, hold the palm ~10–15cm above the camera, fingers spread, and retry.

3) Authenticate (single frame capture)
   python authenticate.py --device 0

Tuning knobs
------------
- In preprocess.py: adjust Gabor thetas/sigmas, CLAHE passes, and thresholding.
- In register.py: QUALITY_MIN and TOP_N.
- In authenticate.py: THRESHOLD (target FAR/FRR).

Notes
-----
- This PC prototype uses classical features (Gabor+LBP). Once stable, replace embeddings with a Siamese CNN and export to TFLite for Raspberry Pi.
- Ensure an IR illuminator (~940nm) and minimal ambient light for best results.
"""
# =========================