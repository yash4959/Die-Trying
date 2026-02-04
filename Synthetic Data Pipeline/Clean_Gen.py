import cv2
import numpy as np
import os
import random
import sys

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_ROOT = 'Staging_Dataset/Clean'

# --- SOURCE FOLDERS ---
SRC_STRIPED = 'Metal_LS_Raw_Imgs'        
SRC_VIA     = 'Via_Arrays_Raw_Imgs'    
SRC_CMP     = 'Dummy_Fill_Raw_Imgs'     

# --- TARGET COUNTS (Optimized for 20% Clean Ratio) ---
# Original Ratio preserved: ~60% Striped, ~13% Via, ~27% CMP
COUNT_STRIPED = 3700 
COUNT_VIA     = 820  
COUNT_CMP     = 1630  

# --- GLOBAL SETTINGS ---
TILE_SIZE = 224
CMP_CANVAS_SIZE = (800, 800)
CMP_TILES_TO_SPLATTER = 400

# ==========================================
# SETUP
# ==========================================
def setup_directories():
    os.makedirs(f'{OUTPUT_ROOT}/images', exist_ok=True)
    os.makedirs(f'{OUTPUT_ROOT}/labels', exist_ok=True)

def save_empty_label(filename):
    txt_path = f'{OUTPUT_ROOT}/labels/{os.path.splitext(filename)[0]}.txt'
    with open(txt_path, 'w') as f:
        pass 

def clean_specific_files(prefix):
    img_dir = f'{OUTPUT_ROOT}/images'
    lbl_dir = f'{OUTPUT_ROOT}/labels'
    if not os.path.exists(img_dir): return
    
    print(f"Cleaning existing '{prefix}' files...")
    removed = 0
    for f in os.listdir(img_dir):
        if f.startswith(prefix):
            os.remove(os.path.join(img_dir, f))
            lbl_path = os.path.join(lbl_dir, f.replace('.png','.txt'))
            if os.path.exists(lbl_path): os.remove(lbl_path)
            removed += 1
    print(f"Removed {removed} old files.")

# ==========================================
# GENERATOR 1: STRIPED
# ==========================================
def rotate_tile_safe(mat, angle):
    if angle == 0: return mat
    h, w = mat.shape[:2]
    big_img = cv2.copyMakeBorder(mat, 2*h, 2*h, 2*w, 2*w, cv2.BORDER_WRAP)
    bh, bw = big_img.shape[:2]
    center = (bw // 2, bh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    nw = int(bh * abs(rot_mat[0,1]) + bw * abs(rot_mat[0,0]))
    nh = int(bh * abs(rot_mat[0,0]) + bw * abs(rot_mat[0,1]))
    rot_mat[0, 2] += nw/2 - center[0]
    rot_mat[1, 2] += nh/2 - center[1]
    rotated = cv2.warpAffine(big_img, rot_mat, (nw, nh))
    rh, rw = rotated.shape[:2]
    sy, sx = (rh // 2) - (h // 2), (rw // 2) - (w // 2)
    return rotated[sy:sy+h, sx:sx+w]

def generate_striped(target_count):
    prefix = "bg_striped"
    clean_specific_files(prefix)
    print(f"--- Generating {target_count} STRIPED backgrounds ---")
    
    if not os.path.exists(SRC_STRIPED): print(f"Missing {SRC_STRIPED}"); return
    files = [f for f in os.listdir(SRC_STRIPED) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not files: print("No images found."); return

    current_count = 0
    angles = [0, 30, 45, 60, 90]

    while current_count < target_count:
        f = random.choice(files)
        img_original = cv2.imread(os.path.join(SRC_STRIPED, f))
        if img_original is None: continue
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        angle = random.choice(angles)
        if angle == 0: img_curr = img_original.copy()
        elif angle == 90: img_curr = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
        else: img_curr = rotate_tile_safe(img_original, angle)

        h, w = img_curr.shape[:2]
        if h < TILE_SIZE or w < TILE_SIZE:
            scale = max(TILE_SIZE/h, TILE_SIZE/w) * 1.1
            img_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
            h, w = img_curr.shape[:2]

        try:
            y = random.randint(0, h - TILE_SIZE)
            x = random.randint(0, w - TILE_SIZE)
            patch = img_curr[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            out_name = f"{prefix}_{current_count:06d}.png"
            cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', patch)
            save_empty_label(out_name)
            current_count += 1
            if current_count % 500 == 0: print(f"  Striped: {current_count}/{target_count}")
        except: continue

# ==========================================
# GENERATOR 2: VIA (PRECISION TILING)
# ==========================================
def generate_via(target_count):
    prefix = "bg_via"
    clean_specific_files(prefix)
    print(f"--- Generating {target_count} VIA backgrounds (Precision Tiling) ---")
    
    if not os.path.exists(SRC_VIA): print(f"Missing {SRC_VIA}"); return
    files = [f for f in os.listdir(SRC_VIA) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not files: print("No images found."); return

    DENSITY_MAP = {
        '2.png': 0.25,  
        '3.png': 0.25,
        '4.png': 0.2,
        '5.png': 0.3,  
        '6.png': 0.3
    }

    current_count = 0
    angles = [0, 30, 45, 60, 90]

    while current_count < target_count:
        f = random.choice(files)
        img_original = cv2.imread(os.path.join(SRC_VIA, f))
        if img_original is None: continue
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        if f in DENSITY_MAP:
            scale = DENSITY_MAP[f]
            img_small = cv2.resize(img_original, (0,0), fx=scale, fy=scale)
            sh, sw = img_small.shape[:2]
            tiles_y = (800 // sh) + 2
            tiles_x = (800 // sw) + 2
            img_curr = np.tile(img_small, (tiles_y, tiles_x, 1))
            angle = random.choice(angles)
            if angle != 0:
                img_curr = rotate_tile_safe(img_curr, angle)
        else:
            angle = random.choice(angles)
            if angle == 0: img_curr = img_original.copy()
            elif angle == 90: img_curr = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
            else: img_curr = rotate_tile_safe(img_original, angle)

        h, w = img_curr.shape[:2]
        if h < TILE_SIZE or w < TILE_SIZE:
            scale = max(TILE_SIZE/h, TILE_SIZE/w) * 1.1
            img_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
            h, w = img_curr.shape[:2]

        try:
            y = random.randint(0, h - TILE_SIZE)
            x = random.randint(0, w - TILE_SIZE)
            patch = img_curr[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            out_name = f"{prefix}_{current_count:06d}.png"
            cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', patch)
            save_empty_label(out_name)
            current_count += 1
            if current_count % 200 == 0: print(f"  Via: {current_count}/{target_count}")
        except: continue

# ==========================================
# GENERATOR 3: DUMMY/CMP
# ==========================================
def apply_sensor_jitter(img):
    beta = random.randint(-20, 20)
    alpha = random.uniform(0.9, 1.1)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def generate_cmp(target_count):
    prefix = "bg_cmp"
    clean_specific_files(prefix)
    print(f"--- Generating {target_count} CMP/DUMMY backgrounds ---")
    
    if not os.path.exists(SRC_CMP): print(f"Missing {SRC_CMP}"); return
    files = [f for f in os.listdir(SRC_CMP) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    safe_tiles = [cv2.imread(os.path.join(SRC_CMP, f)) for f in files]
    safe_tiles = [img for img in safe_tiles if img is not None]
    
    if not safe_tiles: print("No safe tiles found."); return

    current_count = 0
    while current_count < target_count:
        theme_img = random.choice(safe_tiles)
        theme_h, theme_w = theme_img.shape[:2]
        avg_color = np.mean(theme_img, axis=(0,1))
        canvas = np.full((CMP_CANVAS_SIZE[0], CMP_CANVAS_SIZE[1], 3), avg_color, dtype=np.uint8)
        
        for _ in range(CMP_TILES_TO_SPLATTER):
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

            max_y = CMP_CANVAS_SIZE[0] - crop_size
            max_x = CMP_CANVAS_SIZE[1] - crop_size
            pos_y = random.randint(0, max_y)
            pos_x = random.randint(0, max_x)
            canvas[pos_y:pos_y+crop_size, pos_x:pos_x+crop_size] = tile

        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        crop_x = random.randint(0, CMP_CANVAS_SIZE[1] - TILE_SIZE)
        crop_y = random.randint(0, CMP_CANVAS_SIZE[0] - TILE_SIZE)
        final_img = canvas[crop_y:crop_y+TILE_SIZE, crop_x:crop_x+TILE_SIZE]
        
        final_img = apply_sensor_jitter(final_img)
        noise = np.random.normal(0, 5, final_img.shape).astype(np.int16)
        final_img = cv2.add(final_img.astype(np.int16), noise)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        out_name = f"{prefix}_{current_count:06d}.png"
        cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', final_img)
        save_empty_label(out_name)
        current_count += 1
        if current_count % 500 == 0: print(f"  CMP: {current_count}/{target_count}")

# ==========================================
# MAIN MENU
# ==========================================
if __name__ == "__main__":
    setup_directories()
    
    print("\n--- BALANCED BACKGROUND GENERATOR (20% Ratio) ---")
    print(f"Target: {COUNT_STRIPED + COUNT_VIA + COUNT_CMP} Clean Images")
    print("1. Striped / Metal LS")
    print("2. Dummy / CMP")
    print("3. Via Arrays")
    print("4. ALL OF THEM (Recommended)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        generate_striped(COUNT_STRIPED)
    elif choice == '2':
        generate_cmp(COUNT_CMP)
    elif choice == '3':
        generate_via(COUNT_VIA)
    elif choice == '4':
        generate_striped(COUNT_STRIPED)
        generate_cmp(COUNT_CMP)
        generate_via(COUNT_VIA)
    else:
        print("Invalid choice.")
    
    print("\nDone.")