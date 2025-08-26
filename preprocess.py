import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import cosine

# --- Utility: quality metrics ---
def laplacian_sharpness(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def snr_estimate(gray: np.ndarray) -> float:
    # crude SNR proxy: mean/std
    g = gray.astype(np.float32)
    std = g.std() + 1e-6
    return float(g.mean() / std)

# --- Denoise ---
def denoise_gray(gray: np.ndarray) -> np.ndarray:
    # median filter preserves edges; light Gaussian smooth to suppress sensor speckle
    m = cv2.medianBlur(gray, 3)
    g = cv2.GaussianBlur(m, (3,3), 0.6)
    return g

# --- Hand segmentation and ROI estimation (based on convexity defects between finger gaps) ---
def find_hand_contour(gray: np.ndarray) -> np.ndarray:
    # adaptive threshold + morphology to isolate hand
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 5)
    # invert (hand darker @ 940nm backgrounds vary); choose larger area as hand either way
    inv = 255 - thr
    # remove lower 30% (wrist glare)
    h = gray.shape[0]
    inv[int(h*0.7):, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return cnt


def roi_from_contour(gray: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    # Convex hull/defects to find the two finger-gap points
    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        # fallback: bounding-rect centered crop
        x,y,w,h = cv2.boundingRect(cnt)
        side = int(min(w,h) * 0.9)
        cx, cy = x + w//2, y + int(h*0.45)
        x0 = max(0, cx - side//2); y0 = max(0, cy - side//2)
        return gray[y0:y0+side, x0:x0+side]

    # collect valley points
    valleys = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])  # valley point
        valleys.append(far)
    if len(valleys) < 2:
        x,y,w,h = cv2.boundingRect(cnt)
        side = int(min(w,h) * 0.9)
        cx, cy = x + w//2, y + int(h*0.45)
        x0 = max(0, cx - side//2); y0 = max(0, cy - side//2)
        return gray[y0:y0+side, x0:x0+side]

    # pick lowest and third-lowest by y
    valleys_sorted = sorted(valleys, key=lambda p: p[1])  # y increasing (top→bottom)
    p_low = valleys_sorted[-1]
    p_low3 = valleys_sorted[-3] if len(valleys_sorted) >= 3 else valleys_sorted[0]

    # line between p_low and p_low3 defines baseline
    dx = p_low3[0] - p_low[0]
    dy = p_low3[1] - p_low[1]
    angle = np.degrees(np.arctan2(dy, dx))
    # width ~ distance between points
    width = int(np.hypot(dx, dy))
    width = max(width, int(0.35 * gray.shape[1]))

    # center point slightly below the midpoint
    mx, my = int((p_low3[0] + p_low[0]) / 2), int((p_low3[1] + p_low[1]) / 2)
    shift = int(0.6 * width)
    # shift downward along the normal to baseline
    nx = -dy; ny = dx
    nlen = np.hypot(nx, ny) + 1e-6
    cx = int(mx + (nx / nlen) * shift)
    cy = int(my + (ny / nlen) * shift)

    side = int(width)

    # get rotation matrix to align baseline horizontal
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR)

    x0 = max(0, cx - side//2); y0 = max(0, cy - side//2)
    roi = rotated[y0:y0+side, x0:x0+side]
    if roi.size == 0:
        return None
    return roi

# --- Enhancement: equalize → Gabor → CLAHE → threshold ---
def equalize(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def gabor_bank(img: np.ndarray, thetas: int = 8, sigmas=(2.0, 3.0), lambd=8.0, gamma=0.4) -> np.ndarray:
    accum = np.zeros_like(img, dtype=np.float32)
    for th in range(thetas):
        theta = th * np.pi / thetas
        for sigma in sigmas:
            ksize = 5
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(img, cv2.CV_32F, kern)
            accum += resp
    accum /= (thetas * len(sigmas))
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return accum


def clahe_multi(img: np.ndarray, passes: int = 5) -> np.ndarray:
    out = img.copy()
    for i in range(passes):
        clip = 2.0
        grid = 4 + 2*i  # 4,6,8,10,12
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
        out = clahe.apply(out)
    return out


def final_threshold(img: np.ndarray) -> np.ndarray:
    # adaptive threshold to highlight veins
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    # clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

# --- Feature embedding: LBP histogram (compact, robust) ---
def lbp_embedding(img: np.ndarray, P: int = 8, R: int = 2) -> np.ndarray:
    # expect binary mask or enhanced image; we run on enhanced grayscale (pre-threshold) for texture info
    # Ensure non-empty
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins+1), range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist  # dim = P+2

# Combined pipeline for a single frame → (quality, enhanced, mask, embedding)
def process_frame(bgr: np.ndarray, output_size: int = 256):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = denoise_gray(gray)
    q_sharp = laplacian_sharpness(gray)
    q_snr = snr_estimate(gray)

    cnt = find_hand_contour(gray)
    if cnt is None:
        return None
    roi = roi_from_contour(gray, cnt)
    if roi is None or roi.shape[0] < 40 or roi.shape[1] < 40:
        return None

    roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    eq = equalize(roi)
    gab = gabor_bank(eq)
    enh = clahe_multi(gab, passes=5)
    mask = final_threshold(enh)

    # quality gate: require enough vein pixels and sharpness
    vein_ratio = float(mask.mean() / 255.0)
    quality = 0.5 * (q_sharp / 200.0) + 0.5 * (q_snr / 5.0) + 0.5 * (vein_ratio)  # heuristic score

    if vein_ratio < 0.02:  # too sparse
        return None

    emb = lbp_embedding(enh, P=8, R=2)  # 10-dim
    # enrich with a second scale
    emb2 = lbp_embedding(enh, P=16, R=3)  # 18-dim
    embedding = np.concatenate([emb, emb2]).astype(np.float32)  # 28-dim vector

    return {
        "quality": float(quality),
        "roi": roi,
        "enhanced": enh,
        "mask": mask,
        "embedding": embedding,
    }
