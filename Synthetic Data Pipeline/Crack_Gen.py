import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
SAFE_FOLDER = 'Dummy_Fill_Raw_Imgs'          
DIRECT_SOURCE = 'Crack_Raw_Imgs'       
OUTPUT_BASE = 'Staging_Dataset' # Updated folder name

SCRATCH_CLASS_ID = 3
TARGET_SIZE = (224, 224)
IMAGES_TO_GENERATE = 2500  # Increased from 1500 to 2500

CANVAS_SIZE = (800, 800)
TILES_TO_SPLATTER = 400    

os.makedirs(f'{OUTPUT_BASE}/Crack/images', exist_ok=True)
os.makedirs(f'{OUTPUT_BASE}/Crack/labels', exist_ok=True)

# 1. LOAD RESOURCES
safe_tiles = []
if os.path.exists(SAFE_FOLDER):
    for f in os.listdir(SAFE_FOLDER):
        img = cv2.imread(os.path.join(SAFE_FOLDER, f))
        if img is not None: safe_tiles.append(img)

defect_pool = []
if os.path.exists(DIRECT_SOURCE):
    for f in os.listdir(DIRECT_SOURCE):
        img = cv2.imread(os.path.join(DIRECT_SOURCE, f), cv2.IMREAD_UNCHANGED)
        if img is not None: defect_pool.append(img)

if not safe_tiles or not defect_pool:
    print("ERROR: Resources missing. Check your input folders.")
    exit()

# --- UTILITY FUNCTIONS ---
def rotate_image(image, angle):
    """Rotates the crack patch by any angle (0-360)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    border_val = (255, 255, 255, 0) if image.shape[2] == 4 else (255, 255, 255)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

def blend_defect_visible(base_roi, patch):
    """High-contrast blending for visibility on gray background."""
    ph, pw = patch.shape[:2]
    rh, rw = base_roi.shape[:2]
    if (ph != rh) or (pw != rw):
        patch = cv2.resize(patch, (rw, rh))

    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        patch_rgb = cv2.merge([b, g, r])
        mask = a
    else:
        patch_rgb = patch
        gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    alpha_norm = mask.astype(np.float32) / 255.0
    alpha_norm = np.stack([alpha_norm]*3, axis=2)
    
    foreground = patch_rgb.astype(np.float32)
    background = base_roi.astype(np.float32)
    
    blended = (foreground * alpha_norm) + (background * (1.0 - alpha_norm))
    return np.clip(blended, 0, 255).astype(np.uint8)

def generate_coherent_mosaic(idx):
    theme_img = random.choice(safe_tiles)
    h_orig, w_orig = theme_img.shape[:2]
    canvas = np.full((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), np.mean(theme_img, axis=(0,1)), dtype=np.uint8)
    
    # Generate Background Mosaic
    for _ in range(TILES_TO_SPLATTER):
        max_limit = min(h_orig, w_orig, 150)
        min_limit = min(40, max_limit)
        if max_limit <= 5: continue 
        
        crop_size = random.randint(min_limit, max_limit)
        sy, sx = random.randint(0, h_orig - crop_size), random.randint(0, w_orig - crop_size)
        tile = theme_img[sy:sy+crop_size, sx:sx+crop_size].copy()
        tile = np.rot90(tile, random.randint(0, 3))
        py, px = random.randint(0, CANVAS_SIZE[0] - crop_size), random.randint(0, CANVAS_SIZE[1] - crop_size)
        canvas[py:py+crop_size, px:px+crop_size] = tile

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # Defect Injection (90% One crack, 10% Two cracks)
    num_defects = 1 if random.random() < 0.90 else 2
    defect_bboxes = [] 
    
    for _ in range(num_defects):
        patch = random.choice(defect_pool).copy()
        # Full 360 Rotation
        patch = rotate_image(patch, random.randint(0, 360))
        # Small-Medium Scale
        scale = random.uniform(0.12, 0.22) 
        new_w, new_h = int(patch.shape[1] * scale), int(patch.shape[0] * scale)
        patch = cv2.resize(patch, (new_w, new_h))
        
        # Random Splatting on Canvas
        y_s = random.randint(50, CANVAS_SIZE[0] - new_h - 50)
        x_s = random.randint(50, CANVAS_SIZE[1] - new_w - 50)
        
        roi = canvas[y_s:y_s+new_h, x_s:x_s+new_w]
        canvas[y_s:y_s+new_h, x_s:x_s+new_w] = blend_defect_visible(roi, patch)
        defect_bboxes.append((x_s, y_s, new_w, new_h))

    # Random Off-Center Cropping
    target = random.choice(defect_bboxes)
    off_x, off_y = random.randint(-80, 80), random.randint(-80, 80)
    
    cx = max(0, min(target[0] + target[2]//2 - TARGET_SIZE[1]//2 + off_x, CANVAS_SIZE[1] - TARGET_SIZE[1]))
    cy = max(0, min(target[1] + target[3]//2 - TARGET_SIZE[0]//2 + off_y, CANVAS_SIZE[0] - TARGET_SIZE[0]))
    
    final_img = canvas[cy:cy+TARGET_SIZE[0], cx:cx+TARGET_SIZE[1]]
    
    # Save Output
    img_name = f"defect_{idx:05d}"
    cv2.imwrite(f'{OUTPUT_BASE}/Scratch/images/{img_name}.png', final_img)
    with open(f'{OUTPUT_BASE}/Scratch/labels/{img_name}.txt', 'w') as f:
        for (gx, gy, gw, gh) in defect_bboxes:
            nx, ny = gx - cx, gy - cy
            if 0 <= nx < TARGET_SIZE[1] and 0 <= ny < TARGET_SIZE[0]:
                f.write(f"{SCRATCH_CLASS_ID} {(nx+gw/2)/224:.6f} {(ny+gh/2)/224:.6f} {gw/224:.6f} {gh/224:.6f}\n")

print(f"Generating {IMAGES_TO_GENERATE} high-quality images...")
for i in range(IMAGES_TO_GENERATE):
    generate_coherent_mosaic(i)
    if i % 250 == 0: print(f"Progress: {i}/{IMAGES_TO_GENERATE}")
print("Done!")