import cv2
import numpy as np
import random
import os
import math
from scipy.signal import find_peaks

# --- CONFIG ---
RAW_FULL_FOLDER = 'Metal_LS_Raw_Imgs'       
RAW_CROP_FOLDER = 'Metal_LS_Cropped_Lines'  
BASE_IMGS_FOLDER = 'Metal_LS_Base_Imgs'     
FINAL_OUTPUT_DIR = 'Staging_Dataset/Bridge'

TARGET_COUNT = 2500             
TILE_SIZE = 224
ANGLES = [0, 30, 45, 60, 90]

os.makedirs(BASE_IMGS_FOLDER, exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/images', exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/labels', exist_ok=True)

# --- HELPER: TEXTURE LOADER ---
def get_mapped_texture(src_id, type_suffix, target_w, target_h):
    for ext in ['.png', '.jpg']:
        path = os.path.join(RAW_CROP_FOLDER, f"{src_id}{type_suffix}{ext}")
        if os.path.exists(path):
            tex = cv2.imread(path)
            if tex is not None:
                # Crop 2px to remove artifacts
                if tex.shape[0] > 4 and tex.shape[1] > 4:
                    tex = tex[2:-2, 2:-2]
                
                h, w = tex.shape[:2]
                if h > target_h and w > target_w:
                    y = random.randint(0, h - target_h)
                    x = random.randint(0, w - target_w)
                    return tex[y:y+target_h, x:x+target_w]
                else:
                    return cv2.resize(tex, (target_w, target_h))
    return np.random.randint(100, 200, (target_h, target_w, 3), dtype=np.uint8)

# --- SHAPE DRAWING (SCALED UP) ---
def draw_organic_blob(mask, center, radius):
    cx, cy = center
    pts = []
    for i in range(8):
        angle = (i / 8) * 2 * np.pi
        r_var = radius * random.uniform(0.7, 1.3)
        px = int(cx + r_var * np.cos(angle))
        py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

def draw_dendrite(mask, p1, p2, thickness):
    thickness = min(thickness, 4)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    steps = max(10, int(dist / 3))
    points = [p1]
    for i in range(1, steps):
        t = i / steps
        x = int(p1[0] * (1-t) + p2[0] * t)
        y = int(p1[1] * (1-t) + p2[1] * t)
        jitter = random.randint(-1, 1)
        if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]): y += jitter
        else: x += jitter
        points.append((x, y))
    points.append(p2)
    for i in range(len(points)-1):
        cv2.line(mask, points[i], points[i+1], 255, thickness)

def draw_blob_bridge(mask, p1, p2, thickness):
    core_thick = max(2, min(6, int(thickness * 0.8)))
    cv2.line(mask, p1, p2, 255, core_thick)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    steps = max(3, int(dist / 8))
    for i in range(steps + 1):
        t = i / steps
        x = int(p1[0] * (1-t) + p2[0] * t)
        y = int(p1[1] * (1-t) + p2[1] * t)
        blob_r = max(3, min(7, int(thickness * random.uniform(0.8, 1.2))))
        jx = x + random.randint(-1, 1) 
        jy = y + random.randint(-1, 1)
        draw_organic_blob(mask, (jx, jy), blob_r)
    mask[:] = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask[:] = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

def draw_meniscus_bridge(mask, p1, p2, thickness):
    x1, y1 = p1; x2, y2 = p2
    vec_x, vec_y = x2 - x1, y2 - y1
    length = np.hypot(vec_x, vec_y)
    if length == 0: return
    ux, uy = vec_x / length, vec_y / length
    px, py = -uy, ux
    w_anchor = max(4, int(thickness * 1.2))
    w_waist = max(2, int(thickness * 0.6))
    s1_dist = length * 0.25; s2_dist = length * 0.75
    sx1 = x1 + ux * s1_dist; sy1 = y1 + uy * s1_dist
    sx2 = x1 + ux * s2_dist; sy2 = y1 + uy * s2_dist
    pts = []
    pts.append((int(x1 + px * w_anchor), int(y1 + py * w_anchor)))
    pts.append((int(sx1 + px * w_waist), int(sy1 + py * w_waist)))
    pts.append((int(sx2 + px * w_waist), int(sy2 + py * w_waist)))
    pts.append((int(x2 + px * w_anchor), int(y2 + py * w_anchor)))
    pts.append((int(x2 - px * w_anchor), int(y2 - py * w_anchor)))
    pts.append((int(sx2 - px * w_waist), int(sy2 - py * w_waist)))
    pts.append((int(sx1 - px * w_waist), int(sy1 - py * w_waist)))
    pts.append((int(x1 - px * w_anchor), int(y1 - py * w_anchor)))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    core_thickness = max(2, min(8, int(thickness * 0.8)))
    cv2.line(mask, p1, p2, 255, core_thickness)
    k = 3
    mask[:] = cv2.GaussianBlur(mask, (k, k), 0)
    _, mask[:] = cv2.threshold(mask, 110, 255, cv2.THRESH_BINARY)

# --- BACKGROUND GENERATION (FIXED TILING) ---
def rotate_tile_safe(mat, angle):
    """Tiles 5x5 to guarantee no white corners on rotation."""
    if angle == 0: return mat
    h, w = mat.shape[:2]
    
    # FIX: 5x5 Grid (Padding 2*h and 2*w)
    big_img = cv2.copyMakeBorder(mat, 2*h, 2*h, 2*w, 2*w, cv2.BORDER_WRAP)
    
    bh, bw = big_img.shape[:2]
    center = (bw // 2, bh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos, abs_sin = abs(rot_mat[0,0]), abs(rot_mat[0,1])
    nw = int(bh * abs_sin + bw * abs_cos)
    nh = int(bh * abs_cos + bw * abs_sin)
    rot_mat[0, 2] += nw/2 - center[0]
    rot_mat[1, 2] += nh/2 - center[1]
    
    # Rotate the huge canvas
    rotated = cv2.warpAffine(big_img, rot_mat, (nw, nh))
    
    rh, rw = rotated.shape[:2]
    sy, sx = (rh // 2) - (h // 2), (rw // 2) - (w // 2)
    
    # Crop the center
    return rotated[sy:sy+h, sx:sx+w]

def generate_backgrounds_if_needed():
    # 1. Check existing count
    # NOTE: You MUST delete the 'Metal_LS_Base_Imgs' folder manually before running this 
    # if you want to force a full regeneration with the new logic.
    if os.path.exists(BASE_IMGS_FOLDER) and len(os.listdir(BASE_IMGS_FOLDER)) >= TARGET_COUNT:
        print(f"Backgrounds already exist. Skipping generation.")
        return 

    print(f"--- Generating Backgrounds (Balanced: ~71 per image) ---")
    files = [f for f in os.listdir(RAW_FULL_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("Error: No raw images found!")
        return

    num_files = len(files)
    # Calculate exact quotas
    base_quota = TARGET_COUNT // num_files      # e.g. 2500 // 35 = 71
    remainder = TARGET_COUNT % num_files        # e.g. 2500 % 35 = 15
    
    total_generated = 0

    for i, f in enumerate(files):
        # First 'remainder' files get 1 extra image to hit exact target
        file_quota = base_quota + (1 if i < remainder else 0)
        
        src_id = os.path.splitext(f)[0]
        img_original = cv2.imread(os.path.join(RAW_FULL_FOLDER, f))
        
        if img_original is None: 
            print(f"Warning: Could not load {f}")
            continue
            
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        crops_done = 0
        attempts = 0
        
        # Force loop until quota is met for THIS specific image
        while crops_done < file_quota:
            attempts += 1
            if attempts > file_quota * 5: # Safety break if image is too small to crop
                print(f"Warning: Could not generate enough unique crops for {f}")
                break

            # 1. Random Angle
            angle = random.choice(ANGLES)
            if angle == 0: img_curr = img_original.copy()
            elif angle == 90: img_curr = cv2.rotate(img_original, cv2.ROTATE_90_CLOCKWISE)
            else: img_curr = rotate_tile_safe(img_original, angle)

            # 2. Scale Check
            h, w = img_curr.shape[:2]
            if h < TILE_SIZE or w < TILE_SIZE:
                scale = max(TILE_SIZE/h, TILE_SIZE/w) * 1.1
                img_curr = cv2.resize(img_curr, (0,0), fx=scale, fy=scale)
                h, w = img_curr.shape[:2]

            # 3. Random Crop
            try:
                y = random.randint(0, h - TILE_SIZE)
                x = random.randint(0, w - TILE_SIZE)
                patch = img_curr[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                out_name = f"bg_{total_generated:06d}_src{src_id}_deg{angle}.png"
                cv2.imwrite(os.path.join(BASE_IMGS_FOLDER, out_name), patch)
                
                crops_done += 1
                total_generated += 1
            except:
                continue
        
        print(f"  Processed {f}: Generated {crops_done} crops.")

    print(f"Total Backgrounds Generated: {total_generated}")

# --- MAIN BRIDGE GENERATOR ---
def generate_bridges():
    print("--- Generating THICK BRIDGES ---")
    bg_files = [f for f in os.listdir(BASE_IMGS_FOLDER) if f.startswith("bg_")]
    if not bg_files:
        print("Error: No backgrounds!")
        return

    count = 0
    while count < TARGET_COUNT:
        filename = random.choice(bg_files)
        try: src_id = filename.split('_')[2].replace('src', '')
        except: continue

        canvas = cv2.imread(os.path.join(BASE_IMGS_FOLDER, filename))
        if canvas is None: continue
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        is_horz_lines = np.mean(np.abs(gy)) > np.mean(np.abs(gx))
        
        profile = np.mean(gray, axis=1) if is_horz_lines else np.mean(gray, axis=0)
        
        # 1. FIND LINES (Peaks) & GAPS (Valleys)
        line_peaks, _ = find_peaks(profile, height=np.mean(profile), distance=15, prominence=5)
        gap_valleys, _ = find_peaks(-profile, distance=15, prominence=5)
        
        if len(line_peaks) < 2 or len(gap_valleys) < 1: continue 
        
        avg_pitch = np.mean(np.diff(line_peaks))

        # PROBABILITY
        r_num = random.random()
        if r_num < 0.90: num_bridges = 1
        elif r_num < 0.98: num_bridges = 2
        else: num_bridges = 3
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        bridges_drawn = 0
        
        for _ in range(num_bridges):
            for attempt in range(10): 
                
                safe_gaps = [g for g in gap_valleys if 40 < g < 184]
                if not safe_gaps: break
                
                gap_loc = random.choice(safe_gaps)
                
                left_candidates = line_peaks[line_peaks < gap_loc]
                right_candidates = line_peaks[line_peaks > gap_loc]
                
                if len(left_candidates) == 0 or len(right_candidates) == 0: continue
                
                line_1 = left_candidates[-1] 
                line_2 = right_candidates[0] 
                
                dist = line_2 - line_1
                if dist > (avg_pitch * 1.6) or dist > 65: 
                    continue 
                
                coord_var = random.randint(40, 184) 
                
                if mask[gap_loc, coord_var] > 0: continue 

                slant = int(dist * random.uniform(-0.1, 0.1))

                if is_horz_lines: 
                    p1 = (coord_var + slant, line_1)
                    p2 = (coord_var - slant, line_2)
                else: 
                    p1 = (line_1, coord_var + slant)
                    p2 = (line_2, coord_var - slant)

                # Diagonal Fix: Real distance check
                real_dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                if real_dist > 65: continue

                shape_roll = random.random()
                
                # THICKNESS: 15-35% of pitch. Min 2px, Max 10px.
                base_thick = int(avg_pitch * random.uniform(0.15, 0.35))
                thick = max(2, min(10, base_thick)) 
                
                if shape_roll < 0.33:
                    draw_dendrite(mask, p1, p2, thick)
                elif shape_roll < 0.66:
                    draw_blob_bridge(mask, p1, p2, thick)
                else:
                    draw_meniscus_bridge(mask, p1, p2, thick)
                
                bridges_drawn += 1
                break

        if bridges_drawn == 0: continue

        tex = get_mapped_texture(src_id, '_m', 224, 224)
        tex = cv2.rotate(tex, cv2.ROTATE_90_CLOCKWISE) 
        tex = cv2.resize(tex, (224, 224))
        
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        canvas = (canvas.astype(float) * (1 - alpha) + tex.astype(float) * alpha).astype(np.uint8)

        out_name = f"bridge_{count:06d}.png"
        cv2.imwrite(f"{FINAL_OUTPUT_DIR}/images/{out_name}", canvas)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        with open(f"{FINAL_OUTPUT_DIR}/labels/{out_name.replace('.png', '.txt')}", 'w') as f:
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < 2 or h < 2: continue
                cx, cy = (x + w/2)/224, (y + h/2)/224
                nw, nh = w/224, h/224
                f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        count += 1
        if count % 100 == 0: print(f"Bridges: {count}/{TARGET_COUNT}")

if __name__ == "__main__":
    generate_backgrounds_if_needed()
    generate_bridges()