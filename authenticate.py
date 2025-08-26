import cv2
import numpy as np
from models import init_db, User, Template
from preprocess import process_frame
from utils import emb_from_bytes, cosine_sim

THRESHOLD = 0.85  # tune on your validation data


def load_templates(session):
    tpls = session.query(Template).all()
    users = {u.id: u for u in session.query(User).all()}
    items = []
    for t in tpls:
        emb = emb_from_bytes(t.embedding)
        items.append((t.user_id, emb, users.get(t.user_id)))
    return items


def authenticate_once(device=0):
    Session = init_db()
    session = Session()
    items = load_templates(session)
    if not items:
        print("[ERR] No registered users/templates found.")
        return None
    print("Live feed started… Please place your palm in view.")
    cap = cv2.VideoCapture("http://192.0.0.4:8080/video") 
    assert cap.isOpened(), "Camera not accessible"
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[ERR] Camera frame capture failed.")
        return None

    out = process_frame(frame)
    if out is None:
        print("[ERR] Palm not detected or poor quality. Try again.")
        return None

    probe = out["embedding"]
    best_uid, best_user, best_sim = None, None, -1.0
    for uid, emb, user in items:
        sim = cosine_sim(probe, emb)
        if sim > best_sim:
            best_sim = sim
            best_uid, best_user = uid, user

    if best_sim >= THRESHOLD:
        print(f"[OK] Authenticated as {best_user.name} (user_id={best_uid}) — similarity={best_sim:.3f}")
        return best_uid
    else:
        print(f"[FAIL] No match. Best similarity={best_sim:.3f} (threshold={THRESHOLD}).")
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    authenticate_once(args.device)