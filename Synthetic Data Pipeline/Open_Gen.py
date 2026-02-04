import cv2
import numpy as np
import random
import os
import math
from scipy.signal import find_peaks

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FULL_FOLDER = 'Metal_LS_Raw_Imgs'       
RAW_CROP_FOLDER = 'Metal_LS_Cropped_Lines'  
BASE_IMGS_FOLDER = 'Metal_LS_Base_Imgs'     
FINAL_OUTPUT_DIR = 'Staging_Dataset/Open'

TARGET_COUNT = 2500             
TILE_SIZE = 224
ANGLES = [0, 30, 45, 60, 90]

# Ensure folders exist
os.makedirs(BASE_IMGS_FOLDER, exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/images', exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/labels', exist_ok=True)

# ==========================================
# HELPER: TEXTURE LOADER (WITH BRIGHTNESS FIX)
# ==========================================
def get_mapped_texture(src_id, type_suffix, target_w, target_h, target_val_stats=None):
    """
    Loads matching texture.
    CRITICAL UPDATE: Takes 'target_val_stats' (mean, std) of the GAP in the current image.
    It forces the loaded texture to match that darkness.
    """
    tex = None
    # Try finding the source crop
    for ext in ['.png', '.jpg']:
        path = os.path.join(RAW_CROP_FOLDER, f"{src_id}{type_suffix}{ext}")
        if os.path.exists(path):
            tex = cv2.imread(path)
            break
            
    # Fallback if file missing
    if tex is None:
        tex = np.random.randint(20, 50, (target_h, target_w, 3), dtype=np.uint8)

    # 1. Random Crop logic
    if tex is not None:
        if tex.shape[0] > 4 and tex.shape[1] > 4:
            tex = tex[2:-2, 2:-2] # Trim artifacts
        
        h, w = tex.shape[:2]
        if h > target_h and w > target_w:
            y = random.randint(0, h - target_h)
            x = random.randint(0, w - target_w)
            tex = tex[y:y+target_h, x:x+target_w]
        else:
            tex = cv2.resize(tex, (target_w, target_h))

    # 2. BRIGHTNESS CORRECTION (The Fix)
    if target_val_stats is not None:
        target_mean, target_std = target_val_stats
        
        # Convert texture to float
        tex_f = tex.astype(float)
        current_mean = np.mean(tex_f)
        
        # Shift histogram: New = Old - (OldMean - TargetMean)
        # We want the texture to be SLIGHTLY darker than the gap average to look like a deep void
        desired_mean = max(0, target_mean - 10) 
        
        diff = current_mean - desired_mean
        tex_f = tex_f - diff
        
        # Add some noise to prevent flat gray look
        noise = np.random.normal(0, 5, tex_f.shape)
        tex_f = tex_f + noise
        
        tex = np.clip(tex_f, 0, 255).astype(np.uint8)

    return tex

# ==========================================
# PHASE 1: BACKGROUND GENERATOR (Standard)
# ==========================================
def rotate_tile_safe(mat, angle):
    if angle == 0: return mat
    h, w = mat.shape[:2]
    big_img = cv2.copyMakeBorder(mat, 2*h, 2*h, 2*w, 2*w, cv2.BORDER_WRAP)
    bh, bw = big_img.shape[:2]
    center = (bw // 2, bh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos, abs_sin = abs(rot_mat[0,0]), abs(rot_mat[0,1])
    nw = int(bh * abs_sin + bw * abs_cos)
    nh = int(bh * abs_cos + bw * abs_sin)
    rot_mat[0, 2] += nw/2 - center[0]
    rot_mat[1, 2] += nh/2 - center[1]
    rotated = cv2.warpAffine(big_img, rot_mat, (nw, nh))
    rh, rw = rotated.shape[:2]
    sy, sx = (rh // 2) - (h // 2), (rw // 2) - (w // 2)
    return rotated[sy:sy+h, sx:sx+w]

def generate_backgrounds_if_needed():
    if os.path.exists(BASE_IMGS_FOLDER) and len(os.listdir(BASE_IMGS_FOLDER)) >= TARGET_COUNT:
        print(f"Backgrounds already exist. Skipping generation.")
        return 

    print(f"--- Generating Backgrounds (Balanced) ---")
    files = [f for f in os.listdir(RAW_FULL_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not files: return

    num_files = len(files)
    base_quota = TARGET_COUNT // num_files
    remainder = TARGET_COUNT % num_files
    total_generated = 0

    for i, f in enumerate(files):
        file_quota = base_quota + (1 if i < remainder else 0)
        src_id = os.path.splitext(f)[0]
        img_original = cv2.imread(os.path.join(RAW_FULL_FOLDER, f))
        if img_original is None: continue
        if len(img_original.shape) == 2: img_original = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)

        crops_done = 0
        attempts = 0
        while crops_done < file_quota:
            attempts += 1
            if attempts > file_quota * 5: break
            
            angle = random.choice(ANGLES)
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
                out_name = f"bg_{total_generated:06d}_src{src_id}_deg{angle}.png"
                cv2.imwrite(os.path.join(BASE_IMGS_FOLDER, out_name), patch)
                crops_done += 1
                total_generated += 1
            except: continue
        print(f"  Processed {f}: Generated {crops_done} crops.")
    print(f"Total Backgrounds Generated: {total_generated}")

# ==========================================
# PHASE 2: OPEN DEFECT SHAPES
# ==========================================

def draw_clean_cut_poly(mask, start_pt, end_pt, thickness):
    thickness = max(3, min(12, int(thickness)))
    half_t = thickness / 2.0
    x1, y1 = start_pt; x2, y2 = end_pt
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0: return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux 
    
    c1 = (int(x1 + px * half_t), int(y1 + py * half_t))
    c2 = (int(x2 + px * half_t), int(y2 + py * half_t))
    c3 = (int(x2 - px * half_t), int(y2 - py * half_t))
    c4 = (int(x1 - px * half_t), int(y1 - py * half_t))
    
    cv2.fillPoly(mask, [np.array([c1, c2, c3, c4], np.int32)], 255)

def draw_jagged_scratch(mask, center, length, is_horz, thickness):
    cx, cy = center
    thickness = max(1, min(3, int(thickness)))
    slant = int(length * random.uniform(-0.1, 0.1))
    if is_horz:
        p1 = (cx - length//2, cy + slant)
        p2 = (cx + length//2, cy - slant)
    else:
        p1 = (cx + slant, cy - length//2)
        p2 = (cx - slant, cy + length//2)
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    steps = max(5, int(dist / 3))
    points = [p1]
    for i in range(1, steps):
        t = i / steps
        x = int(p1[0] * (1-t) + p2[0] * t)
        y = int(p1[1] * (1-t) + p2[1] * t)
        jitter = random.randint(-1, 1)
        if is_horz: y += jitter
        else: x += jitter
        points.append((x, y))
    points.append(p2)
    for i in range(len(points)-1):
        cv2.line(mask, points[i], points[i+1], 255, thickness)

def draw_solid_mousebite(mask, edge_point, line_width):
    cx, cy = edge_point
    radius = max(4, min(12, int(line_width * 0.4))) 
    num_pts = 8
    pts = []
    for i in range(num_pts):
        angle = (i / num_pts) * 2 * np.pi
        r_var = radius * random.uniform(0.8, 1.2)
        px = int(cx + r_var * np.cos(angle))
        py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

# ==========================================
# MAIN OPEN GENERATOR
# ==========================================
def generate_opens():
    print("--- Generating OPENS (Texture Corrected) ---")
    bg_files = [f for f in os.listdir(BASE_IMGS_FOLDER) if f.startswith("bg_")]
    if not bg_files:
        print("Error: No backgrounds found in Base_Imgs!")
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
        
        peaks, _ = find_peaks(profile, height=np.mean(profile), distance=15, prominence=5)
        valleys, _ = find_peaks(-profile, distance=15, prominence=5)
        
        safe_peaks = [p for p in peaks if 40 < p < 184]
        if not safe_peaks: continue
        
        # --- MEASURE GAP DARKNESS ---
        # We find the darkest 20% of pixels in the image to estimate the "Gap Color"
        # This is more robust than using valley indices which might be noisy
        darkest_pixels = gray[gray < np.percentile(gray, 20)]
        if len(darkest_pixels) > 0:
            gap_mean = np.mean(darkest_pixels)
            gap_std = np.std(darkest_pixels)
        else:
            gap_mean = 30 # Fallback
            gap_std = 5
            
        r_num = random.random()
        if r_num < 0.90: num_opens = 1
        elif r_num < 0.98: num_opens = 2
        else: num_opens = 3
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        opens_drawn = 0
        
        for _ in range(num_opens):
            for attempt in range(10):
                line_loc = random.choice(safe_peaks)
                left_valleys = valleys[valleys < line_loc]
                right_valleys = valleys[valleys > line_loc]
                
                if len(left_valleys) == 0 or len(right_valleys) == 0: continue
                
                v_left = left_valleys[-1]
                v_right = right_valleys[0]
                line_width = v_right - v_left
                
                if line_width > 55: continue 
                
                coord_var = random.randint(40, 184)
                if mask[line_loc, coord_var] > 0: continue
                
                open_type = random.random()
                
                if open_type < 0.65: # CLEAN CUT
                    thick = int(line_width * random.uniform(0.25, 0.55))
                    slant = int(line_width * random.uniform(-0.1, 0.1))
                    if is_horz_lines: 
                        p1 = (coord_var + slant, v_left)
                        p2 = (coord_var - slant, v_right)
                    else: 
                        p1 = (v_left, coord_var + slant)
                        p2 = (v_right, coord_var - slant)
                    draw_clean_cut_poly(mask, p1, p2, thick)

                elif open_type < 0.90: # SCRATCH
                    length = int(line_width * 1.1)
                    thick = random.randint(1, 3)
                    center = (coord_var, line_loc) if is_horz_lines else (line_loc, coord_var)
                    draw_jagged_scratch(mask, center, length, not is_horz_lines, thick)
                
                else: # MOUSEBITE
                    target_valley = v_left if random.random() < 0.5 else v_right
                    direction_vector = target_valley - line_loc
                    offset = int(direction_vector * 0.40) 
                    edge_pos = line_loc + offset
                    if is_horz_lines: center = (coord_var, edge_pos)
                    else: center = (edge_pos, coord_var)
                    draw_solid_mousebite(mask, center, line_width)
                
                opens_drawn += 1
                break

        if opens_drawn == 0: continue

        # --- APPLY TEXTURE WITH BRIGHTNESS CORRECTION ---
        # Pass the gap statistics we measured earlier!
        tex = get_mapped_texture(src_id, '_b', 224, 224, target_val_stats=(gap_mean, gap_std))
        
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        canvas = (canvas.astype(float) * (1 - alpha) + tex.astype(float) * alpha).astype(np.uint8)

        out_name = f"open_{count:06d}.png"
        cv2.imwrite(f"{FINAL_OUTPUT_DIR}/images/{out_name}", canvas)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        with open(f"{FINAL_OUTPUT_DIR}/labels/{out_name.replace('.png', '.txt')}", 'w') as f:
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w < 2 or h < 2: continue
                cx, cy = (x + w/2)/224, (y + h/2)/224
                nw, nh = w/224, h/224
                f.write(f"1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        
        count += 1
        if count % 100 == 0: print(f"Opens: {count}/{TARGET_COUNT}")

if __name__ == "__main__":
    generate_backgrounds_if_needed()
    generate_opens()