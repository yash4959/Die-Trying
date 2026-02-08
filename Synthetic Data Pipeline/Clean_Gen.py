import cv2
import numpy as np
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_ROOT = 'Staging_Dataset/Clean'

# --- SOURCE FOLDERS ---
SRC_STRIPED = 'Metal_LS_Raw_Imgs'        
SRC_VIA     = 'Via_Arrays_Raw_Imgs'    
SRC_CMP     = 'Dummy_Fill_Raw_Imgs'     

# --- TARGET COUNTS ---
TARGET_COUNT_PER_TYPE = 500

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

# ==========================================
# HELPER: ROTATION
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

# ==========================================
# GENERATOR 1: METAL LS (STRIPED)
# ==========================================
def generate_metal_ls():
    prefix = "clean_metal_ls"
    print(f"--- Generating {TARGET_COUNT_PER_TYPE} Metal LS backgrounds ---")
    
    if not os.path.exists(SRC_STRIPED): print(f"Missing {SRC_STRIPED}"); return
    files = [f for f in os.listdir(SRC_STRIPED) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: print("No images found."); return

    current_count = 0
    angles = [0, 30, 45, 60, 90]

    while current_count < TARGET_COUNT_PER_TYPE:
        f = random.choice(files)
        img_original = cv2.imread(os.path.join(SRC_STRIPED, f))
        if img_original is None: continue
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        # Random Rotate
        angle = random.choice(angles)
        if angle == 0: img_curr = img_original.copy()
        elif angle == 90: img_curr = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
        else: img_curr = rotate_tile_safe(img_original, angle)

        h, w = img_curr.shape[:2]
        # Resize if too small
        if h < TILE_SIZE or w < TILE_SIZE:
            scale = max(TILE_SIZE/h, TILE_SIZE/w) * 1.1
            img_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
            h, w = img_curr.shape[:2]

        try:
            y = random.randint(0, h - TILE_SIZE)
            x = random.randint(0, w - TILE_SIZE)
            patch = img_curr[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # --- NAMING FIX ---
            out_name = f"{prefix}_{current_count:05d}.png"
            
            cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', patch)
            save_empty_label(out_name)
            current_count += 1
            if current_count % 100 == 0: print(f"  Metal LS: {current_count}/{TARGET_COUNT_PER_TYPE}")
        except: continue

# ==========================================
# GENERATOR 2: VIA ARRAYS (TILING)
# ==========================================
def generate_via_arrays():
    prefix = "clean_via_array"
    print(f"--- Generating {TARGET_COUNT_PER_TYPE} Via Array backgrounds ---")
    
    if not os.path.exists(SRC_VIA): print(f"Missing {SRC_VIA}"); return
    files = [f for f in os.listdir(SRC_VIA) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: print("No images found."); return

    current_count = 0
    angles = [0, 30, 45, 60, 90]

    while current_count < TARGET_COUNT_PER_TYPE:
        f = random.choice(files)
        img_original = cv2.imread(os.path.join(SRC_VIA, f))
        if img_original is None: continue
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        # Scale down slightly to ensure dense patterns
        scale = random.uniform(0.2, 0.4)
        img_small = cv2.resize(img_original, (0,0), fx=scale, fy=scale)
        sh, sw = img_small.shape[:2]
        
        # Tile it to fill a buffer larger than TILE_SIZE
        tiles_y = (400 // sh) + 2
        tiles_x = (400 // sw) + 2
        img_tiled = np.tile(img_small, (tiles_y, tiles_x, 1))
        
        # Rotate
        angle = random.choice(angles)
        if angle == 0: img_curr = img_tiled
        elif angle == 90: img_curr = cv2.rotate(img_tiled, cv2.ROTATE_90_CLOCKWISE)
        else: img_curr = rotate_tile_safe(img_tiled, angle)

        h, w = img_curr.shape[:2]
        if h < TILE_SIZE or w < TILE_SIZE: continue

        try:
            y = random.randint(0, h - TILE_SIZE)
            x = random.randint(0, w - TILE_SIZE)
            patch = img_curr[y:y+TILE_SIZE, x:x+TILE_SIZE]
            
            # --- NAMING FIX ---
            out_name = f"{prefix}_{current_count:05d}.png"
            
            cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', patch)
            save_empty_label(out_name)
            current_count += 1
            if current_count % 100 == 0: print(f"  Via Array: {current_count}/{TARGET_COUNT_PER_TYPE}")
        except: continue

# ==========================================
# GENERATOR 3: DUMMY CMP (MOSAIC)
# ==========================================
def apply_sensor_jitter(img):
    beta = random.randint(-20, 20)
    alpha = random.uniform(0.9, 1.1)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def generate_dummy_cmp():
    prefix = "clean_dummy_cmp"
    print(f"--- Generating {TARGET_COUNT_PER_TYPE} Dummy CMP backgrounds ---")
    
    if not os.path.exists(SRC_CMP): print(f"Missing {SRC_CMP}"); return
    files = [f for f in os.listdir(SRC_CMP) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Load all tiles into memory for speed
    safe_tiles = []
    for f in files:
        img = cv2.imread(os.path.join(SRC_CMP, f))
        if img is not None: safe_tiles.append(img)
    
    if not safe_tiles: print("No safe tiles found."); return

    current_count = 0
    while current_count < TARGET_COUNT_PER_TYPE:
        # Base canvas color
        theme_img = random.choice(safe_tiles)
        theme_h, theme_w = theme_img.shape[:2]
        avg_color = np.mean(theme_img, axis=(0,1))
        canvas = np.full((CMP_CANVAS_SIZE[0], CMP_CANVAS_SIZE[1], 3), avg_color, dtype=np.uint8)
        
        # Splatter tiles
        for _ in range(CMP_TILES_TO_SPLATTER):
            crop_size = random.randint(80, 150)
            
            # Get random tile
            src = random.choice(safe_tiles)
            sh, sw = src.shape[:2]
            
            if sh > crop_size and sw > crop_size:
                sy = random.randint(0, sh - crop_size)
                sx = random.randint(0, sw - crop_size)
                tile = src[sy:sy+crop_size, sx:sx+crop_size].copy()
            else:
                tile = cv2.resize(src, (crop_size, crop_size))

            # Augment Tile
            k = random.randint(0, 3)
            tile = np.rot90(tile, k)
            if random.random() > 0.5: tile = cv2.flip(tile, 1)

            # Paste
            max_y = CMP_CANVAS_SIZE[0] - crop_size
            max_x = CMP_CANVAS_SIZE[1] - crop_size
            pos_y = random.randint(0, max_y)
            pos_x = random.randint(0, max_x)
            canvas[pos_y:pos_y+crop_size, pos_x:pos_x+crop_size] = tile

        # Final Polish
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        crop_x = random.randint(0, CMP_CANVAS_SIZE[1] - TILE_SIZE)
        crop_y = random.randint(0, CMP_CANVAS_SIZE[0] - TILE_SIZE)
        final_img = canvas[crop_y:crop_y+TILE_SIZE, crop_x:crop_x+TILE_SIZE]
        
        # Sensor Noise
        final_img = apply_sensor_jitter(final_img)
        noise = np.random.normal(0, 5, final_img.shape).astype(np.int16)
        final_img = cv2.add(final_img.astype(np.int16), noise)
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)

        # Save
        out_name = f"{prefix}_{current_count:05d}.png"
        cv2.imwrite(f'{OUTPUT_ROOT}/images/{out_name}', final_img)
        save_empty_label(out_name)
        
        current_count += 1
        if current_count % 100 == 0: print(f"  Dummy CMP: {current_count}/{TARGET_COUNT_PER_TYPE}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    setup_directories()
    
    print("\n--- CLEAN BACKGROUND GENERATOR ---")
    print(f"Target: 500 per class ({TARGET_COUNT_PER_TYPE * 3} Total)")
    
    # Run all
    generate_metal_ls()
    generate_via_arrays()
    generate_dummy_cmp()
    
    print("\nAll clean images generated successfully.")