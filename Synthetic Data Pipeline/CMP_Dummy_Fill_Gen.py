import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
SAFE_FOLDER = 'Dummy_Fill_Raw_Imgs'              # Background Patterns
INJECT_SOURCE = 'CMP_Mined_Patches'        # Raw defects
DIRECT_SOURCE = 'CMP_Manual_Patches'        # Texture Patches
OUTPUT_BASE = 'Staging_Dataset'

SCRATCH_CLASS_ID = 3
TARGET_SIZE = (224, 224)
IMAGES_TO_GENERATE = 1500  

# CANVAS SETTINGS
CANVAS_SIZE = (800, 800)
TILES_TO_SPLATTER = 400    

# QC SETTINGS
JITTER_SCALE_MIN = 0.8     
JITTER_SCALE_MAX = 1.0     

os.makedirs(f'{OUTPUT_BASE}/CMP/images', exist_ok=True)
os.makedirs(f'{OUTPUT_BASE}/CMP/labels', exist_ok=True)

# 1. LOAD RESOURCES
print("Loading resources...")
safe_tiles = []
if os.path.exists(SAFE_FOLDER):
    for f in os.listdir(SAFE_FOLDER):
        img = cv2.imread(os.path.join(SAFE_FOLDER, f))
        if img is not None: safe_tiles.append(img)

# LOAD DEFECTS
defect_pool = []
if os.path.exists(DIRECT_SOURCE):
    for f in os.listdir(DIRECT_SOURCE):
        img = cv2.imread(os.path.join(DIRECT_SOURCE, f), cv2.IMREAD_UNCHANGED)
        if img is not None: defect_pool.append(img)

if os.path.exists(INJECT_SOURCE):
    for f in os.listdir(INJECT_SOURCE):
        img = cv2.imread(os.path.join(INJECT_SOURCE, f)) 
        if img is not None:
            b, g, r = cv2.split(img)
            alpha = np.ones_like(b) * 255
            img = cv2.merge([b, g, r, alpha])
            defect_pool.append(img)

if not safe_tiles:
    print("ERROR: No safe tiles found.")
    exit()

print(f"Stats: {len(safe_tiles)} Safe Tiles | {len(defect_pool)} Total Defect Sources")

# --- HELPER FUNCTIONS ---
def save_label(bboxes, filename):
    txt_path = f'{OUTPUT_BASE}/Scratch/labels/{os.path.splitext(filename)[0]}.txt'
    H, W = TARGET_SIZE
    with open(txt_path, 'w') as f:
        for (x, y, w, h) in bboxes:
            f.write(f"{SCRATCH_CLASS_ID} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")

def apply_sensor_jitter(img):
    beta = random.randint(-20, 20)
    alpha = random.uniform(0.9, 1.1)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# --- THE NATURAL BLENDER (V77) ---
def blend_defect_natural(base_roi, patch):
    """
    Pastes scratch on top, but matches lighting better and avoids
    cartoonish contrast.
    """
    ph, pw = patch.shape[:2]
    rh, rw = base_roi.shape[:2]
    
    # 1. Resize Patch
    if (ph != rh) or (pw != rw):
        patch = cv2.resize(patch, (rw, rh))

    # 2. Extract Components
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        patch_rgb = cv2.merge([b, g, r])
        
        # Determine Mask
        if np.min(a) < 250:
            mask = a
        else:
            # Auto-mask for mined scratches
            gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
            grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
            
            # Lower threshold to capture softer scratch details too
            _, mask = cv2.threshold(grad, 15, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
    else:
        patch_rgb = patch
        mask = np.ones((rh, rw), dtype=np.uint8) * 255

    # 3. MILD CONTRAST BOOST (Fixed from V76)
    # Instead of 2.5x, we use 1.2x (Subtle pop)
    mean_val = np.mean(patch_rgb)
    patch_rgb = cv2.convertScaleAbs(patch_rgb, alpha=1.1, beta=-0.2*mean_val)
    
    # 4. LIGHTING MATCH (Crucial)
    # Shift patch mean to match ROI mean almost fully (90%)
    roi_mean = np.mean(base_roi)
    patch_mean = np.mean(patch_rgb)
    offset = roi_mean - patch_mean
    
    # Apply 90% of the offset (was 50% in V76)
    patch_rgb = np.clip(cv2.add(patch_rgb.astype(np.float32), offset * 0.9), 0, 255).astype(np.uint8)

    # 5. COMPOSITE
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    alpha_norm = mask.astype(np.float32) / 255.0
    alpha_norm = np.stack([alpha_norm]*3, axis=2)
    
    foreground = patch_rgb.astype(np.float32)
    background = base_roi.astype(np.float32)
    
    blended = (foreground * alpha_norm) + (background * (1.0 - alpha_norm))
    
    return np.clip(blended, 0, 255).astype(np.uint8)

def generate_coherent_mosaic(idx):
    # 1. PICK THEME
    theme_img = random.choice(safe_tiles)
    theme_h, theme_w = theme_img.shape[:2]

    # 2. BACKGROUND
    avg_color = np.mean(theme_img, axis=(0,1))
    canvas = np.full((CANVAS_SIZE[0], CANVAS_SIZE[1], 3), avg_color, dtype=np.uint8)
    
    for _ in range(TILES_TO_SPLATTER):
        crop_size = random.randint(80, 150)
        if theme_h > crop_size and theme_w > crop_size:
            sy = random.randint(0, theme_h - crop_size)
            sx = random.randint(0, theme_w - crop_size)
            tile = theme_img[sy:sy+crop_size, sx:sx+crop_size].copy()
        else:
            tile = cv2.resize(theme_img, (crop_size, crop_size))

        k = random.randint(0, 3)
        tile = np.rot90(tile, k)
        if random.random() > 0.5: tile = cv2.flip(tile, 1)

        max_y = CANVAS_SIZE[0] - crop_size
        max_x = CANVAS_SIZE[1] - crop_size
        pos_y = random.randint(0, max_y)
        pos_x = random.randint(0, max_x)
        
        canvas[pos_y:pos_y+crop_size, pos_x:pos_x+crop_size] = tile

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    # 3. INJECT DEFECTS (Natural)
    num_defects = random.choices([1, 2, 3], weights=[0.80, 0.15, 0.05], k=1)[0]
    defect_bboxes = [] 
    
    for _ in range(num_defects):
        if not defect_pool: break
        
        patch = random.choice(defect_pool).copy()
        k = random.randint(0, 3)
        patch = np.rot90(patch, k)
        if random.random() > 0.5: patch = cv2.flip(patch, 1)
        
        scale = random.uniform(JITTER_SCALE_MIN, JITTER_SCALE_MAX)
        ph, pw = patch.shape[:2]
        new_w, new_h = int(pw * scale), int(ph * scale)
        patch = cv2.resize(patch, (new_w, new_h))
        
        margin = 200
        center_y = random.randint(margin, CANVAS_SIZE[0]-margin)
        center_x = random.randint(margin, CANVAS_SIZE[1]-margin)
        
        y_start = center_y - new_h//2
        x_start = center_x - new_w//2
        
        roi = canvas[y_start:y_start+new_h, x_start:x_start+new_w].copy()
        
        # --- BLEND (NATURAL) ---
        final_tile = blend_defect_natural(roi, patch)
        
        canvas[y_start:y_start+new_h, x_start:x_start+new_w] = final_tile
        defect_bboxes.append((x_start, y_start, new_w, new_h))

    # 4. CROP & SAVE
    if defect_bboxes:
        target_gx, target_gy, target_gw, target_gh = random.choice(defect_bboxes)
        target_center_x = target_gx + target_gw // 2
        target_center_y = target_gy + target_gh // 2
        
        crop_x = target_center_x - (TARGET_SIZE[1] // 2)
        crop_y = target_center_y - (TARGET_SIZE[0] // 2)
        crop_x += random.randint(-40, 40)
        crop_y += random.randint(-40, 40)
    else:
        crop_x = random.randint(0, CANVAS_SIZE[1] - TARGET_SIZE[1])
        crop_y = random.randint(0, CANVAS_SIZE[0] - TARGET_SIZE[0])
        
    crop_x = max(0, min(crop_x, CANVAS_SIZE[1] - TARGET_SIZE[1]))
    crop_y = max(0, min(crop_y, CANVAS_SIZE[0] - TARGET_SIZE[0]))
    
    final_img = canvas[crop_y:crop_y+TARGET_SIZE[0], crop_x:crop_x+TARGET_SIZE[1]]
    
    final_bboxes = []
    for (gx, gy, gw, gh) in defect_bboxes:
        new_x = gx - crop_x
        new_y = gy - crop_y
        x1, y1 = max(0, new_x), max(0, new_y)
        x2, y2 = min(TARGET_SIZE[1], new_x + gw), min(TARGET_SIZE[0], new_y + gh)
        if (x2-x1) * (y2-y1) > (gw * gh * 0.2):
             final_bboxes.append((x1, y1, x2-x1, y2-y1))

    final_img = apply_sensor_jitter(final_img)
    noise = np.random.normal(0, 5, final_img.shape).astype(np.int16)
    final_img = cv2.add(final_img.astype(np.int16), noise)
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)

    out_name = f"final_v77_{idx:05d}.png"
    cv2.imwrite(f'{OUTPUT_BASE}/Scratch/images/{out_name}', final_img)
    save_label(final_bboxes, out_name)

print(f"Generating {IMAGES_TO_GENERATE} V77 Natural Mosaics...")
for i in range(IMAGES_TO_GENERATE):
    generate_coherent_mosaic(i)
    if i % 200 == 0: print(f"...{i}")
print("Done.")