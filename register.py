import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF warnings

from absl import logging
logging.set_verbosity(logging.ERROR)       # suppress absl logs

import cv2
import time
import numpy as np
from models import init_db, User, Template
from preprocess import process_frame, laplacian_sharpness
from utils import emb_to_bytes
import mediapipe as mp

# Constants
TOP_N = 12
CAPTURE_SECONDS = 3
QUALITY_MIN = 0.35  # minimum acceptable frame quality

# Mediapipe hands detector
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,          # enable tracking
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def capture_frames(seconds=CAPTURE_SECONDS, device=0, debug=False):
    cap = cv2.VideoCapture(device if device != -1 else "http://192.168.0.4:8080/video")
    assert cap.isOpened(), "Camera not accessible"
    frames = []
    print("Live feed started… Please place your palm in view.")
    t_end = time.time() + seconds

    if debug:
        out = cv2.VideoWriter("capture.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0,
                              (int(cap.get(3)), int(cap.get(4))))

    while time.time() < t_end:
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(frame)

        if debug:
            cv2.imshow("Palm Capture", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if debug:
        out.release()
        cv2.destroyAllWindows()

    return frames

def select_top_frames(frames):
    scored = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        score = laplacian_sharpness(gray)
        scored.append((score, f))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [f for _, f in scored[:TOP_N]]
    return keep

def register_user(name: str, phone: str = None, device=0, debug=False):
    Session = init_db()
    session = Session()

    frames = capture_frames(device=device, debug=debug)
    best_frames = select_top_frames(frames)

    embeddings = []
    qualities = []
    kept = 0

    for f in best_frames:
        frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        if not results.multi_hand_landmarks:
            print("[WARN] No hand detected. Place your hand in view.")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks for debug
        if debug:
            mp.solutions.drawing_utils.draw_landmarks(f, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Hand Detection", f)
            cv2.waitKey(1)

        # Open-palm check: require at least 3 fingers extended
        tips_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        fingers_open = [hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y
                        for tip, pip in zip(tips_ids, pip_ids)]

        if sum(fingers_open) < 3:
            print("[WARN] Fist or partially closed hand detected. Please open your palm.")
            continue

        # Process frame
        out = process_frame(f)
        if out is None or out["quality"] < QUALITY_MIN:
            continue

        embeddings.append(out["embedding"])
        qualities.append(out["quality"])
        kept += 1

    if debug:
        cv2.destroyAllWindows()

    if kept < 4:
        print("[WARN] Registration failed: not enough quality frames. Try again.")
        return None

    E = np.median(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    avg_q = float(np.mean(qualities))

    user = User(name=name, phone=phone)
    session.add(user)
    session.commit()

    tpl = Template(user_id=user.id, embedding=emb_to_bytes(E), dim=E.shape[0],
                   num_frames=kept, quality=avg_q)
    session.add(tpl)
    session.commit()

    print(f"[OK] Registered {name} (user_id={user.id}) with {kept} frames. Quality={avg_q:.3f}")
    return user.id

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--phone", default=None)
    ap.add_argument("--device", type=int, default=-1, help="0 for Pi/USB cam, -1 for IP webcam")
    ap.add_argument("--debug", action="store_true", help="Show live feed and hand landmarks")
    args = ap.parse_args()
    register_user(args.name, args.phone, args.device, debug=args.debug)
