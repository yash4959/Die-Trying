import cv2
import numpy as np
import random
import os
import re
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION
# ==========================================
BASE_IMGS_FOLDER = 'Metal_LS_Base_Imgs'
RAW_CROP_FOLDER = 'Metal_LS_Cropped_Lines'
OUTPUT_BASE = 'Staging_Dataset_Mixed_Final'

TARGET_COUNT = 2500
TILE_SIZE = 224

# Class IDs
BRIDGE_ID = 0
OPEN_ID = 1
CMP_BUMP_ID = 2
LER_ID = 3

os.makedirs(f'{OUTPUT_BASE}/images', exist_ok=True)
os.makedirs(f'{OUTPUT_BASE}/labels', exist_ok=True)

# ==========================================
# GEOMETRY & NOISE UTILITIES
# ==========================================

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, M

def rotate_point(point, M):
    x, y = point
    new_x = (M[0, 0] * x) + (M[0, 1] * y) + M[0, 2]
    new_y = (M[1, 0] * x) + (M[1, 1] * y) + M[1, 2]
    return (new_x, new_y)

def apply_heavy_edge_noise(img, zone_mask, intensity=65):
    """
    Applies high-intensity monochromatic noise specifically to the edges.
    """
    h, w = img.shape[:2]
    # Generate monochromatic grain
    noise = np.random.normal(0, intensity, (h, w)).astype(np.float32)
    img_f = img.astype(np.float32)
    
    if img.ndim == 3:
        for c in range(3):
            img_f[:, :, c] += noise * zone_mask
    else:
        img_f += noise * zone_mask
        
    return np.clip(img_f, 0, 255).astype(np.uint8)

# ==========================================
# DEFECT LOGIC
# ==========================================

def apply_selective_ler(img):
    """
    Applies reduced waviness but heavy grain noise at the edges.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    col_prof = gray.mean(axis=0)
    line_x = np.where(np.abs(np.gradient(col_prof)) > np.percentile(np.abs(np.gradient(col_prof)), 80))[0]
    centers = [int(np.mean(l)) for l in np.split(line_x, np.where(np.diff(line_x) > 4)[0] + 1) if len(l) > 2]
    
    if not centers: return img, None
    cx = random.choice(centers)
    
    # Coordinates for Remap
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = map_x.astype(np.float32); map_y = map_y.astype(np.float32)
    
    # REDUCED WAVINESS: Lower Amplitude (0.4 to 0.9)
    y_idx = np.arange(h)
    jitter = random.uniform(0.4, 0.9) * np.sin(2 * np.pi * y_idx / random.uniform(6, 12))
    
    influence = np.exp(-np.abs(np.arange(w) - cx) / 7)
    map_x += jitter[:, None] * influence
    
    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # INCREASED EDGE NOISE: Localized heavy grain
    edge_map = cv2.convertScaleAbs(cv2.Sobel(cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY), cv2.CV_16S, 1, 0))
    # Thicken edge mask to catch more 'Aju Baju' area
    edge_mask = cv2.threshold(edge_map, 25, 1, cv2.THRESH_BINARY)[1].astype(np.float32)
    edge_mask = cv2.dilate(edge_mask, np.ones((3,3), np.uint8))
    
    distorted = apply_heavy_edge_noise(distorted, edge_mask * influence, intensity=75)
    
    return distorted, {'cx': cx, 'cy': h/2, 'w': 20, 'h': h}

# ==========================================
# MASTER GENERATOR (Summary)
# ==========================================
# ... [Rest of generate_sample remains identical to ensure 2 defects and rotation-awareness] ...

def inject_bridge_or_open(img, dtype, peaks, valleys, src_id):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if dtype == OPEN_ID:
        line = random.choice(peaks)
        v_nearby = sorted([v for v in valleys if abs(v - line) < 20])
        if len(v_nearby) < 2: return img, None
        v1, v2 = v_nearby[0], v_nearby[-1]
        cv2.line(mask, (line, v1), (line, v2), 255, random.randint(8, 14))
        tex_path = os.path.join(RAW_CROP_FOLDER, f"{src_id}_b.png")
    else: # BRIDGE
        valley = random.choice(valleys)
        p_nearby = sorted([p for p in peaks if abs(p - valley) < 20])
        if len(p_nearby) < 2: return img, None
        p1, p2 = p_nearby[0], p_nearby[-1]
        c_v = random.randint(60, 160)
        cv2.line(mask, (p1, c_v), (p2, c_v), 255, 6)
        tex_path = os.path.join(RAW_CROP_FOLDER, f"{src_id}_m.png")

    tex = cv2.imread(tex_path) if os.path.exists(tex_path) else np.random.randint(50, 180, (h,w,3), dtype=np.uint8)
    tex = cv2.resize(tex, (w, h))
    alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float)/255.0
    img = (img.astype(float)*(1-alpha) + tex.astype(float)*alpha).astype(np.uint8)
    x, y, bw, bh = cv2.boundingRect(mask)
    return img, (x + bw/2, y + bh/2, bw, bh)

def generate_sample(idx, bg_files):
    fname = random.choice(bg_files)
    angle = int(re.search(r'deg(\d+)', fname).group(1)) if re.search(r'deg(\d+)', fname) else 0
    src_id = fname.split('_')[2].replace('src', '') if 'src' in fname else "0"
    
    canvas = cv2.imread(os.path.join(BASE_IMGS_FOLDER, fname))
    if canvas is None: return False
    canvas = cv2.resize(canvas, (TILE_SIZE, TILE_SIZE))
    
    vert_img, M_fwd = rotate_image(canvas, -angle)
    gray = cv2.cvtColor(vert_img, cv2.COLOR_BGR2GRAY)
    prof = np.mean(gray, axis=0)
    peaks, _ = find_peaks(prof, height=np.mean(prof), distance=15)
    valleys, _ = find_peaks(-prof, distance=15)
    
    if len(peaks) < 2: return False
    defect_types = random.sample([BRIDGE_ID, OPEN_ID, CMP_BUMP_ID, LER_ID], 2)
    temp_labels = []

    for dtype in defect_types:
        if dtype == LER_ID:
            vert_img, box = apply_selective_ler(vert_img)
            if box: temp_labels.append((LER_ID, (box['cx'], box['cy'], box['w'], box['h']), True)) 
        elif dtype in [BRIDGE_ID, OPEN_ID]:
            vert_img, box = inject_bridge_or_open(vert_img, dtype, peaks, valleys, src_id)
            if box: temp_labels.append((dtype, box, False))
        elif dtype == CMP_BUMP_ID:
            line = random.choice(peaks); c_v = random.randint(60, 160)
            m = np.zeros_like(gray, dtype=np.float32)
            cv2.circle(m, (line, c_v), 6, 1.0, -1)
            m = cv2.GaussianBlur(m, (15,15), 0)
            sob = cv2.Sobel(m, cv2.CV_32F, 1, 1)
            vert_img = np.clip(vert_img.astype(np.float32)*(1.0 + sob[...,None]*0.4), 0, 255).astype(np.uint8)
            temp_labels.append((CMP_BUMP_ID, (line, c_v, 24, 24), False))

    if len(temp_labels) < 2: return False
    final_canvas, M_inv = rotate_image(vert_img, angle)
    
    # Realism: Global Poisson Noise
    final_canvas = (np.random.poisson(final_canvas.astype(np.float32)/255.0 * 45) / 45 * 255).astype(np.uint8)

    yolo_final = []
    for cid, box, is_ler in temp_labels:
        cx, cy, bw, bh = box
        if is_ler:
            pts = [rotate_point(p, M_inv) for p in [(cx-bw/2, 0), (cx+bw/2, 0), (cx+bw/2, 224), (cx-bw/2, 224)]]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            nx, ny, nw, nh = (min(xs)+max(xs))/2/224, (min(ys)+max(ys))/2/224, (max(xs)-min(xs))/224, (max(ys)-min(ys))/224
        else:
            p = rotate_point((cx, cy), M_inv)
            nx, ny, nw, nh = p[0]/224, p[1]/224, bw/224, bh/224
        yolo_final.append(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

    cv2.imwrite(f"{OUTPUT_BASE}/images/final_{idx:05d}.png", final_canvas)
    with open(f"{OUTPUT_BASE}/labels/final_{idx:05d}.txt", 'w') as f:
        f.write("\n".join(yolo_final))
    return True

if __name__ == "__main__":
    bg_files = [f for f in os.listdir(BASE_IMGS_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    count = 0
    print(f"Generating 2500 samples with Heavy Edge Noise and Stochastic LER...")
    while count < TARGET_COUNT:
        if generate_sample(count, bg_files):
            count += 1
            if count % 100 == 0: print(f"Progress: {count}")