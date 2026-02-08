import cv2
import numpy as np
import random
import os
import re
from scipy.signal import find_peaks

# ================= CONFIGURATION =================
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

# ================= TEXTURE UTILS =================

def add_salt_and_pepper(patch, amount=0.08):
    num_salt = np.ceil(amount * patch.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in patch.shape]
    patch[tuple(coords)] = 255
    num_pepper = np.ceil(amount * patch.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in patch.shape]
    patch[tuple(coords)] = 0
    return patch

def roughen_mask_edges(mask):
    h, w = mask.shape
    noise = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    edges = cv2.Canny(mask, 100, 200)
    jitter = cv2.bitwise_and(edges, noise)
    kernel_size = random.randint(3, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if random.random() > 0.5: rough = cv2.erode(mask, kernel, iterations=1)
    else: rough = cv2.dilate(mask, kernel, iterations=1)
    return cv2.addWeighted(mask, 0.4, rough, 0.6, 0)

def apply_global_grit(image):
    """Applies unified SEM-style grit to the final image."""
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image.copy()
    gray = gray.astype(np.float32)
    h, w = gray.shape
    gaussian_noise = np.random.normal(0, 15, (h, w)).astype(np.float32)
    noisy_gray = gray + gaussian_noise
    salt_mask = np.random.random((h, w)) < 0.015
    noisy_gray[salt_mask] = 255
    pepper_mask = np.random.random((h, w)) < 0.015
    noisy_gray[pepper_mask] = 0
    final_gray = np.clip(noisy_gray, 0, 255).astype(np.uint8)
    return cv2.merge([final_gray, final_gray, final_gray])

def get_mapped_texture(src_id, type_suffix, target_w, target_h):
    for ext in ['.png', '.jpg']:
        path = os.path.join(RAW_CROP_FOLDER, f"{src_id}{type_suffix}{ext}")
        if os.path.exists(path):
            tex = cv2.imread(path)
            if tex is not None:
                if tex.shape[0] > 4 and tex.shape[1] > 4: tex = tex[2:-2, 2:-2]
                h, w = tex.shape[:2]
                if h > target_h and w > target_w:
                    y = random.randint(0, h - target_h)
                    x = random.randint(0, w - target_w)
                    patch = tex[y:y+target_h, x:x+target_w]
                else: patch = cv2.resize(tex, (target_w, target_h))
                noise = np.random.normal(0, 15, patch.shape).astype(np.int16)
                patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                return patch
    base = np.random.randint(100, 200, (target_h, target_w, 3), dtype=np.uint8)
    return base

def apply_texture_grafting(clean_bg, dirty_canvas):
    if len(clean_bg.shape) == 3:
        clean_gray = cv2.cvtColor(clean_bg, cv2.COLOR_BGR2GRAY)
        dirty_gray = cv2.cvtColor(dirty_canvas, cv2.COLOR_BGR2GRAY)
    else: clean_gray, dirty_gray = clean_bg, dirty_canvas
    bg_blur = cv2.GaussianBlur(clean_gray, (5, 5), 0)
    real_texture = cv2.subtract(clean_gray, bg_blur)
    thresh_val = np.mean(dirty_gray) * 0.85 
    _, binary_shape = cv2.threshold(dirty_gray, thresh_val, 255, cv2.THRESH_BINARY)
    soft_shape = cv2.GaussianBlur(binary_shape, (3, 3), 0)
    mask_lines = clean_gray > thresh_val
    mask_gaps = clean_gray <= thresh_val
    val_line = np.mean(clean_gray[mask_lines]) if np.any(mask_lines) else 160
    val_gap = np.mean(clean_gray[mask_gaps]) if np.any(mask_gaps) else 40
    norm_shape = soft_shape.astype(float) / 255.0
    base_layer = (val_line * norm_shape + val_gap * (1 - norm_shape)).astype(np.uint8)
    final_result = cv2.add(base_layer, real_texture)
    return cv2.merge([final_result, final_result, final_result])

# ================= ROTATION & COORDINATE MATH (CRITICAL FIX) =================

def get_rotated_line_coords(img, angle_deg):
    """
    1. Un-rotates the image temporarily to make lines vertical.
    2. Finds peaks (lines).
    3. Picks points.
    4. Rotates points back to original angle.
    Returns: p1, p2, thickness
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Un-rotate
    # If lines are at 60 deg, we rotate by -60 to make them vertical (0 deg)
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    straight_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    if len(straight_img.shape) == 3:
        straight_gray = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)
    else:
        straight_gray = straight_img

    # 2. Find Peaks (Center Strip Only to avoid corner artifacts)
    strip_h = 40
    center_strip = straight_gray[cy-strip_h//2 : cy+strip_h//2, :]
    profile = np.mean(center_strip, axis=0)
    
    # Peaks = Bright Lines, Valleys = Dark Gaps
    line_peaks, _ = find_peaks(profile, height=np.mean(profile), distance=15, prominence=5)
    
    if len(line_peaks) < 2: return None, None, None

    # 3. Pick Adjacent Lines
    idx = random.randint(0, len(line_peaks) - 2)
    x1 = line_peaks[idx]
    x2 = line_peaks[idx+1]
    
    avg_pitch = x2 - x1
    
    # Choose random Y along the vertical lines
    # Avoid edges
    y_straight = random.randint(50, h - 50)
    
    # Add slant/jitter in the STRAIGHT frame (before rotation)
    # This simulates finding a spot where lines align
    slant = int(avg_pitch * random.uniform(-0.1, 0.1))
    
    pt1_straight = np.array([x1, y_straight + slant, 1])
    pt2_straight = np.array([x2, y_straight - slant, 1])
    
    # 4. Re-Rotate Points
    # We need the inverse matrix (Rotate by +angle)
    M_inv = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    
    p1 = M_inv.dot(pt1_straight)
    p2 = M_inv.dot(pt2_straight)
    
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    
    thick = max(2, min(10, int(avg_pitch * random.uniform(0.15, 0.40))))
    
    return p1, p2, thick

# ================= VECTOR WARP ENGINE =================

def warp_kissing_lines(img, p1, p2):
    """
    Simulates 'Line Collapse' by pinching ONLY the two lines involved.
    Rotation invariant.
    """
    h, w = img.shape[:2]
    cx, cy = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    
    gap_vec = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=np.float32)
    gap_len = np.linalg.norm(gap_vec)
    if gap_len < 1e-3: return img, None
    
    u_gap = gap_vec / gap_len 
    u_line = np.array([-u_gap[1], u_gap[0]]) 
    
    # TIGHT ROI
    radius_long = int(gap_len * 1.2) 
    radius_across = int(gap_len * 0.55) 
    
    safe_r = max(radius_long, radius_across) + 5
    x_min, x_max = max(0, int(cx - safe_r)), min(w, int(cx + safe_r))
    y_min, y_max = max(0, int(cy - safe_r)), min(h, int(cy + safe_r))
    
    roi_h, roi_w = y_max - y_min, x_max - x_min
    if roi_h <= 0 or roi_w <= 0: return img, None
    
    roi = img[y_min:y_max, x_min:x_max]
    grid_x, grid_y = np.meshgrid(np.arange(roi_w), np.arange(roi_h))
    rel_x = (grid_x + x_min) - cx
    rel_y = (grid_y + y_min) - cy
    
    dist_along = rel_x * u_line[0] + rel_y * u_line[1]
    dist_across = rel_x * u_gap[0] + rel_y * u_gap[1]
    
    norm_along = np.clip(np.abs(dist_along) / radius_long, 0, 1)
    weight_along = np.exp(-3.0 * norm_along**2) 
    norm_across = np.clip(np.abs(dist_across) / radius_across, 0, 1)
    weight_across = np.exp(-1.0 * norm_across**2)

    win_x = 1.0 - (np.abs(grid_x - roi_w/2) / (roi_w/2))**4
    win_y = 1.0 - (np.abs(grid_y - roi_h/2) / (roi_h/2))**4
    boundary_mask = np.clip(win_x * win_y, 0, 1)

    target_shift = (gap_len / 2.0) * 1.02 
    pull_mag = np.sign(dist_across) * target_shift * weight_along * weight_across * boundary_mask
    
    shift_x = pull_mag * u_gap[0]
    shift_y = pull_mag * u_gap[1]
    map_x = (grid_x + shift_x).astype(np.float32)
    map_y = (grid_y + shift_y).astype(np.float32)
    
    warped_roi = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    result = img.copy()
    result[y_min:y_max, x_min:x_max] = warped_roi
    
    box_r = int(gap_len * 1.0)
    bbox = (int(cx - box_r), int(cy - box_r), box_r*2, box_r*2)
    return result, bbox

# ================= SHAPE GENERATORS =================

def draw_organic_blob(mask, center, radius):
    cx, cy = center; pts = []
    num_verts = random.randint(10, 16)
    for i in range(num_verts):
        angle = (i / num_verts) * 2 * np.pi; r_var = radius * random.uniform(0.8, 1.4)
        px = int(cx + r_var * np.cos(angle)); py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

def draw_dendrite(mask, p1, p2, thickness):
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    steps = max(15, int(dist / 2)); points = [p1]
    current = np.array(p1, dtype=float); vector = (np.array(p2)-current)/steps
    for i in range(1, steps):
        base = current + vector
        nxt = (int(base[0]+random.uniform(-2,2)), int(base[1]+random.uniform(-2,2)))
        points.append(nxt); current = base
    points.append(p2)
    for i in range(len(points)-1):
        cv2.line(mask, points[i], points[i+1], 255, max(1, thickness + random.randint(-1, 1)))

def draw_blob_bridge(mask, p1, p2, thickness):
    if random.random() > 0.5: cv2.line(mask, p1, p2, 255, max(1, int(thickness*0.6)))
    dist = np.linalg.norm(np.array(p1)-np.array(p2))
    steps = max(3, int(dist/6))
    for i in range(steps + 1):
        if random.random() > 0.7: continue
        t = i/steps; x = int(p1[0]*(1-t)+p2[0]*t); y = int(p1[1]*(1-t)+p2[1]*t)
        draw_organic_blob(mask, (x, y), max(2, min(8, int(thickness*random.uniform(0.7, 1.5)))))

def draw_chewing_gum_bridge(mask, p1, p2, thickness):
    p1=np.array(p1,dtype=float); p2=np.array(p2,dtype=float)
    vec=p2-p1; length=np.linalg.norm(vec)
    if length==0:return
    unit_vec=vec/length; perp_vec=np.array([-unit_vec[1],unit_vec[0]]) 
    steps=40; top=[]; bot=[]
    bulge=random.uniform(0.1, 0.4)
    for i in range(steps+1):
        t=i/steps; center=p1+vec*t
        center += perp_vec * np.sin(t*np.pi) * random.uniform(-0.05,0.05) * length
        width = thickness * (1.0 + (4*t*(1-t)*bulge)) * random.uniform(0.9,1.1)
        top.append(center + perp_vec*width); bot.append(center - perp_vec*width)
    poly = np.array(top + bot[::-1], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    mask[:] = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask[:] = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# ================= BACKGROUND LOGIC =================

def rotate_tile_safe(mat, angle):
    if angle == 0: return mat
    h, w = mat.shape[:2]
    big_img = cv2.copyMakeBorder(mat, 2*h, 2*h, 2*w, 2*w, cv2.BORDER_REPLICATE)
    bh, bw = big_img.shape[:2]
    center = (bw // 2, bh // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos, abs_sin = abs(rot_mat[0,0]), abs(rot_mat[0,1])
    nw = int(bh * abs_sin + bw * abs_cos)
    nh = int(bh * abs_cos + bw * abs_sin)
    rot_mat[0, 2] += nw/2 - center[0]
    rot_mat[1, 2] += nh/2 - center[1]
    rotated = cv2.warpAffine(big_img, rot_mat, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    rh, rw = rotated.shape[:2]
    sy, sx = (rh // 2) - (h // 2), (rw // 2) - (w // 2)
    return rotated[sy:sy+h, sx:sx+w]

def generate_backgrounds_if_needed():
    if os.path.exists(BASE_IMGS_FOLDER) and len(os.listdir(BASE_IMGS_FOLDER)) >= TARGET_COUNT:
        print(f"Backgrounds already exist. Skipping generation.")
        return 
    print(f"--- Generating Backgrounds ---")
    files = [f for f in os.listdir(RAW_FULL_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
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
        crops_done = 0; attempts = 0
        while crops_done < file_quota:
            attempts += 1
            if attempts > file_quota * 10: break
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
        print(f" Processed {f}: Generated {crops_done} crops.")

# ================= MAIN PIPELINE =================

def generate_bridges():
    print("--- Generating BRIDGES (All Orientations Fixed + Global Grit) ---")
    bg_files = [f for f in os.listdir(BASE_IMGS_FOLDER) if f.startswith("bg_")]
    if not bg_files: return

    count = 0
    while count < TARGET_COUNT:
        filename = random.choice(bg_files)
        try: 
            # Extract Angle and Source ID
            src_id = filename.split('_')[2].replace('src', '')
            angle_str = filename.split('_')[-1].replace('.png','').replace('deg','')
            angle = int(angle_str)
        except: continue
        
        canvas = cv2.imread(os.path.join(BASE_IMGS_FOLDER, filename))
        if canvas is None: continue
        clean_bg_copy = canvas.copy()

        # --- CORRECT COORDINATE FINDING ---
        p1, p2, thick = get_rotated_line_coords(canvas, angle)
        
        if p1 is None: continue

        # Check valid distance
        real_dist = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        if real_dist > 65: continue # Skip if gap too wide
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        bridges_drawn = 0; final_bbox = None; is_kissing_type = False
        
        shape_roll = random.random()
        if shape_roll > 0.60: # 40% Chance Kissing (WARP)
            canvas, final_bbox = warp_kissing_lines(canvas, p1, p2)
            is_kissing_type = True
            bridges_drawn = 1
        else:
            if shape_roll < 0.20: draw_dendrite(mask, p1, p2, thick) 
            elif shape_roll < 0.40: draw_blob_bridge(mask, p1, p2, thick) 
            else: draw_chewing_gum_bridge(mask, p1, p2, thick)
            bridges_drawn = 1

        out_name = f"bridge_{count:06d}.png"
        final_img_pre_noise = None

        if is_kissing_type:
            final_img_pre_noise = canvas
            if final_bbox:
                bx, by, bw, bh = final_bbox
                cx, cy = (bx + bw/2)/224, (by + bh/2)/224
                nw, nh = bw/224, bh/224
                with open(f"{FINAL_OUTPUT_DIR}/labels/{out_name.replace('.png', '.txt')}", 'w') as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        else:
            mask = roughen_mask_edges(mask)
            tex = get_mapped_texture(src_id, '_m', 224, 224)
            tex = cv2.rotate(tex, cv2.ROTATE_90_CLOCKWISE) 
            tex = cv2.resize(tex, (224, 224))
            mask_soft = cv2.GaussianBlur(mask, (3, 3), 0)
            alpha = cv2.cvtColor(mask_soft, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
            dirty_canvas = (canvas.astype(float) * (1 - alpha) + tex.astype(float) * alpha).astype(np.uint8)
            final_img_pre_noise = apply_texture_grafting(clean_bg_copy, dirty_canvas)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(f"{FINAL_OUTPUT_DIR}/labels/{out_name.replace('.png', '.txt')}", 'w') as f:
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    if w < 2 or h < 2: continue
                    cx, cy = (x + w/2)/224, (y + h/2)/224
                    nw, nh = w/224, h/224
                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        # === APPLY GLOBAL GRIT TO EVERYTHING ===
        final_img = apply_global_grit(final_img_pre_noise)

        cv2.imwrite(f"{FINAL_OUTPUT_DIR}/images/{out_name}", final_img)
        count += 1
        if count % 100 == 0: print(f"Bridges: {count}/{TARGET_COUNT}")

if __name__ == "__main__":
    generate_backgrounds_if_needed()
    generate_bridges()