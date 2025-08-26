import cv2
import time
import numpy as np
from models import init_db, User, Template
from preprocess import process_frame, laplacian_sharpness
from utils import emb_to_bytes

TOP_N = 12
CAPTURE_SECONDS = 3
QUALITY_MIN = 0.35  # adjust after testing


def capture_frames(seconds=CAPTURE_SECONDS, device=0, debug=False):
    cap = cv2.VideoCapture("http://192.0.0.4:8080/video")
    assert cap.isOpened(), "Camera not accessible"
    frames = []
    print("Live feed started… Please place your palm in view.")
    t_end = time.time() + seconds

    # Optional: Video writer if you want to save
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


def register_user(name: str, phone: str = None, device=0):
    Session = init_db()
    session = Session()

    frames = capture_frames()
    best = select_top_frames(frames)

    embeddings = []
    qualities = []
    kept = 0

    for f in best:
        out = process_frame(f)
        if out is None:
            continue
        if out["quality"] < QUALITY_MIN:
            continue
        embeddings.append(out["embedding"])
        qualities.append(out["quality"])
        kept += 1

    if kept < 4:
        print("[WARN] Registration failed: not enough quality frames. Try again with better lighting/position.")
        return None

    # median embedding is robust
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
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Show live feed and save video for debugging")
    args = ap.parse_args()
    register_user(args.name, args.phone, args.device)
