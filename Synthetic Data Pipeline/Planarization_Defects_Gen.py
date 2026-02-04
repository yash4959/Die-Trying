import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
SAFE_FOLDER = 'Dummy_Fill_Raw_Imgs' 
CMP_INJECT_SOURCE = 'CMP_Mined_Patches'
CMP_DIRECT_SOURCE = 'CMP_Manual_Patches'
CRACK_SOURCE = 'Crack_Raw_Imgs'
OUTPUT_BASE = 'Staging_Dataset/Planarization defects'

CMP_CLASS_ID = 3
CRACK_CLASS_ID = 4 
TARGET_SIZE = (224, 224)
IMAGES_TO_GENERATE = 2500 

CANVAS_SIZE = (800, 800)
TILES_TO_SPLATTER = 400 

os.makedirs(f'{OUTPUT_BASE}/images', exist_ok=True)
os.makedirs(f'{OUTPUT_BASE}/labels', exist_ok=True)

# --- 1. LOAD RESOURCES ---
safe_tiles = []
if os.path.exists(SAFE_FOLDER):
    for f in os.listdir(SAFE_FOLDER):
        img = cv2.imread(os.path.join(SAFE_FOLDER, f))
        if img is not None: safe_tiles.append(img)

cmp_pool = []
for src in [CMP_DIRECT_SOURCE, CMP_INJECT_SOURCE]:
    if os.path.exists(src):
        for f in os.listdir(src):
            img = cv2.imread(os.path.join(src, f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.shape[2] == 3 and src == CMP_INJECT_SOURCE:
                    b, g, r = cv2.split(img)
                    alpha = np.ones_like(b) * 255
                    img = cv2.merge([b, g, r, alpha])
                cmp_pool.append(img)

crack_pool = []
if os.path.exists(CRACK_SOURCE):
    for f in os.listdir(CRACK_SOURCE):
        img = cv2.imread(os.path.join(CRACK_SOURCE, f), cv2.IMREAD_UNCHANGED)
        if img is not None: crack_pool.append(img)

# --- 2. LOGIC FUNCTIONS (CMP & CRACK) ---

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    border_val = (255, 255, 255, 0) if image.shape[2] == 4 else (255, 255, 255)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def blend_cmp_natural(base_roi, patch):
    ph, pw = patch.shape[:2]
    rh, rw = base_roi.shape[:2]
    if (ph != rh) or (pw != rw): patch = cv2.resize(patch, (rw, rh))
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        patch_rgb = cv2.merge([b, g, r])
        if np.min(a) < 250: mask = a
        else:
            gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
            _, mask = cv2.threshold(grad, 15, 255, cv2.THRESH_BINARY)
            mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
    else:
        patch_rgb, mask = patch, np.ones((rh, rw), dtype=np.uint8) * 255
    patch_rgb = cv2.convertScaleAbs(patch_rgb, alpha=1.1, beta=-0.2*np.mean(patch_rgb))
    offset = np.mean(base_roi) - np.mean(patch_rgb)
    patch_rgb = np.clip(cv2.add(patch_rgb.astype(np.float32), offset * 0.9), 0, 255).astype(np.uint8)
    alpha_norm = np.stack([cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0]*3, axis=2)
    return ((patch_rgb.astype(np.float32) * alpha_norm) + (base_roi.astype(np.float32) * (1.0 - alpha_norm))).astype(np.uint8)

def blend_crack_visible(base_roi, patch):
    ph, pw = patch.shape[:2]
    rh, rw = base_roi.shape[:2]
    if (ph != rh) or (pw != rw): patch = cv2.resize(patch, (rw, rh))
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        patch_rgb, mask = cv2.merge([b, g, r]), a
    else:
        patch_rgb = patch
        _, mask = cv2.threshold(cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    alpha_norm = np.stack([cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0]*3, axis=2)
    return ((patch_rgb.astype(np.float32) * alpha_norm) + (base_roi.astype(np.float32) * (1.0 - alpha_norm))).astype(np.uint8)

# --- 3. GENERATION ---

def generate_sample(idx):
    # Determine defect counts (Minimum 2 total)
    roll = random.random()
    if roll < 0.80:    # 80% Both (1 CMP + 1 Crack)
        num_cmp, num_crack = 1, 1
    elif roll < 0.90:  # 10% Single type but DOUBLE count (2 CMP or 2 Crack)
        num_cmp, num_crack = (2, 0) if random.random() < 0.5 else (0, 2)
    else:              # 10% Triple (1+2 or 2+1)
        num_cmp, num_crack = (1, 2) if random.random() < 0.5 else (2, 1)

    # Background Creation
    theme_img = random.choice(safe_tiles)
    canvas = np.full((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), np.mean(theme_img, axis=(0,1)), dtype=np.uint8)
    h_orig, w_orig = theme_img.shape[:2]
    for _ in range(TILES_TO_SPLATTER):
        size = random.randint(min(40, min(h_orig, w_orig)), min(150, min(h_orig, w_orig)))
        sy, sx = random.randint(0, h_orig - size), random.randint(0, w_orig - size)
        tile = np.rot90(theme_img[sy:sy+size, sx:sx+size], random.randint(0, 3))
        py, px = random.randint(0, CANVAS_SIZE[0] - size), random.randint(0, CANVAS_SIZE[1] - size)
        canvas[py:py+size, px:px+size] = tile
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    bboxes = []
    # To ensure defects are close enough to be in the same 224x224 crop:
    spawn_x, spawn_y = random.randint(200, 450), random.randint(200, 450)

    for _ in range(num_cmp):
        p = random.choice(cmp_pool).copy()
        p = np.rot90(p, random.randint(0, 3))
        s = random.uniform(0.8, 1.0)
        nw, nh = int(p.shape[1] * s), int(p.shape[0] * s)
        p = cv2.resize(p, (nw, nh))
        xs, ys = spawn_x + random.randint(-60, 60), spawn_y + random.randint(-60, 60)
        roi = canvas[ys:ys+nh, xs:xs+nw].copy()
        canvas[ys:ys+nh, xs:xs+nw] = blend_cmp_natural(roi, p)
        bboxes.append((CMP_CLASS_ID, xs, ys, nw, nh))

    for _ in range(num_crack):
        p = random.choice(crack_pool).copy()
        p = rotate_image(p, random.randint(0, 360))
        s = random.uniform(0.15, 0.25)
        nw, nh = int(p.shape[1] * s), int(p.shape[0] * s)
        p = cv2.resize(p, (nw, nh))
        xs, ys = spawn_x + random.randint(-60, 60), spawn_y + random.randint(-60, 60)
        roi = canvas[ys:ys+nh, xs:xs+nw]
        canvas[ys:ys+nh, xs:xs+nw] = blend_crack_visible(roi, p)
        bboxes.append((CRACK_CLASS_ID, xs, ys, nw, nh))

    # Center crop on the "spawn zone" to capture multiple defects
    cx = max(0, min(spawn_x + 30 - 112 + random.randint(-20, 20), CANVAS_SIZE[1] - 224))
    cy = max(0, min(spawn_y + 30 - 112 + random.randint(-20, 20), CANVAS_SIZE[0] - 224))
    
    final_img = canvas[cy:cy+224, cx:cx+224]
    
    out_name = f"min2_{idx:05d}"
    with open(f'{OUTPUT_BASE}/labels/{out_name}.txt', 'w') as f:
        count = 0
        for (cid, gx, gy, gw, gh) in bboxes:
            nx, ny = gx - cx, gy - cy
            x1, y1 = max(0, nx), max(0, ny)
            x2, y2 = min(224, nx + gw), min(224, ny + gh)
            if (x2 > x1) and (y2 > y1):
                f.write(f"{cid} {(x1+(x2-x1)/2)/224:.6f} {(y1+(y2-y1)/2)/224:.6f} {(x2-x1)/224:.6f} {(y2-y1)/224:.6f}\n")
                count += 1
    
    if count >= 2: # Verify minimum 2 defects were actually captured in the crop
        cv2.imwrite(f'{OUTPUT_BASE}/images/{out_name}.png', final_img)
    else:
        return False # Signal to retry if crop missed the second defect
    return True

print(f"Generating {IMAGES_TO_GENERATE} images with MINIMUM 2 defects...")
success_count = 0
while success_count < IMAGES_TO_GENERATE:
    if generate_sample(success_count):
        success_count += 1
        if success_count % 100 == 0: print(f"Progress: {success_count}/{IMAGES_TO_GENERATE}")

print("Done.")