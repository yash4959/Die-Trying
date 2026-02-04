import cv2
import numpy as np
import os

# --- CONFIG ---
INPUT_DIR = 'Carinthia_Imgs' 
OUTPUT_DIR = 'CMP_Mined_Patches'
CROP_SIZE = 120                 

# SENSITIVITY SETTINGS
# 1. Variance (Roughness): Lower = More sensitive to "bumpy" scratches
VAR_THRESH = 50       

# 2. Edges (Sharpness): Lower = More sensitive to "faint lines"
CANNY_LOW = 30        # Edge detection lower bound
CANNY_HIGH = 100      # Edge detection upper bound
MIN_EDGE_PIXELS = 20  # How many "scratch pixels" need to be found to save?

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))]
print(f"Deep-Mining {len(files)} images...")

count = 0
saved = 0
for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(path)
    if img is None: continue
    
    h, w = img.shape[:2]
    
    # 1. CROP CENTER
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - CROP_SIZE // 2)
    y2 = min(h, cy + CROP_SIZE // 2)
    x1 = max(0, cx - CROP_SIZE // 2)
    x2 = min(w, cx + CROP_SIZE // 2)
    
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: continue

    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # TEST 1: VARIANCE (Is it rough?)
    variance = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
    is_rough = variance > VAR_THRESH

    # TEST 2: EDGES (Is there a line?)
    edges = cv2.Canny(gray_crop, CANNY_LOW, CANNY_HIGH)
    edge_pixels = np.count_nonzero(edges)
    is_sharp = edge_pixels > MIN_EDGE_PIXELS
    
    # DECISION: Keep if EITHER is true
    if is_rough or is_sharp:
        cv2.imwrite(f'{OUTPUT_DIR}/scratch_{saved:05d}.png', crop)
        saved += 1
    
    count += 1
    if count % 500 == 0: print(f"Scanned {count}... Saved {saved} so far.")

print(f"Success. Saved {saved} scratches (High Sensitivity Mode).")