import cv2
import numpy as np
import random
import os
from scipy.signal import find_peaks

# ================= CONFIGURATION =================
RAW_FULL_FOLDER = 'Metal_LS_Raw_Imgs'       
RAW_CROP_FOLDER = 'Metal_LS_Cropped_Lines'  
BASE_IMGS_FOLDER = 'Metal_LS_Base_Imgs'     
FINAL_OUTPUT_DIR = 'Staging_Dataset/Open'

TARGET_COUNT = 2500             
TILE_SIZE = 224
ANGLES = [0, 30, 45, 60, 90]

os.makedirs(BASE_IMGS_FOLDER, exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/images', exist_ok=True)
os.makedirs(f'{FINAL_OUTPUT_DIR}/labels', exist_ok=True)

# ================= TEXTURE & NOISE UTILS =================

def apply_global_grit(image):
    """
    Applies unified SEM-style grit to the final image.
    ADJUSTED: Much lower S&P noise to avoid 'TV Static' look.
    """
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image.copy()
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    # 1. Gaussian Grain (Reduced from 15 -> 10)
    # This provides the base "texture" without fuzziness
    gaussian_noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
    noisy_gray = gray + gaussian_noise
    
    # 2. Salt & Pepper (Drastically Reduced from 1.5% -> 0.2%)
    # Now represents rare hot pixels/pitting rather than snow
    salt_mask = np.random.random((h, w)) < 0.002
    noisy_gray[salt_mask] = 255
    pepper_mask = np.random.random((h, w)) < 0.002
    noisy_gray[pepper_mask] = 0
    
    final_gray = np.clip(noisy_gray, 0, 255).astype(np.uint8)
    return cv2.merge([final_gray, final_gray, final_gray])

def roughen_mask_edges(mask):
    """Makes the cut look jagged and etched, not geometric."""
    h, w = mask.shape
    noise = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    edges = cv2.Canny(mask, 100, 200)
    jitter = cv2.bitwise_and(edges, noise)
    
    kernel_size = random.randint(3, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if random.random() > 0.5: rough = cv2.erode(mask, kernel, iterations=1)
    else: rough = cv2.dilate(mask, kernel, iterations=1)
    
    return cv2.addWeighted(mask, 0.4, rough, 0.6, 0)

def get_mapped_texture(src_id, type_suffix, target_w, target_h, target_val_stats=None):
    """Loads gap texture and adjusts brightness to match the real gaps."""
    tex = None
    for ext in ['.png', '.jpg']:
        path = os.path.join(RAW_CROP_FOLDER, f"{src_id}{type_suffix}{ext}")
        if os.path.exists(path):
            tex = cv2.imread(path)
            break
            
    if tex is None:
        tex = np.random.randint(20, 50, (target_h, target_w, 3), dtype=np.uint8)

    # Random Crop
    if tex is not None:
        if tex.shape[0] > 4 and tex.shape[1] > 4: tex = tex[2:-2, 2:-2]
        h, w = tex.shape[:2]
        if h > target_h and w > target_w:
            y = random.randint(0, h - target_h)
            x = random.randint(0, w - target_w)
            tex = tex[y:y+target_h, x:x+target_w]
        else:
            tex = cv2.resize(tex, (target_w, target_h))

    # Brightness Matching (Critical for Opens to look deep)
    if target_val_stats is not None:
        target_mean, target_std = target_val_stats
        tex_f = tex.astype(float)
        current_mean = np.mean(tex_f)
        # Target slightly darker than average gap to look like a clean break
        desired_mean = max(0, target_mean - 5) 
        diff = current_mean - desired_mean
        tex_f = tex_f - diff
        tex = np.clip(tex_f, 0, 255).astype(np.uint8)

    return tex

def apply_texture_grafting(clean_bg, dirty_canvas):
    """Melds the gap texture into the line using frequency separation."""
    if len(clean_bg.shape) == 3:
        clean_gray = cv2.cvtColor(clean_bg, cv2.COLOR_BGR2GRAY)
        dirty_gray = cv2.cvtColor(dirty_canvas, cv2.COLOR_BGR2GRAY)
    else: clean_gray, dirty_gray = clean_bg, dirty_canvas

    bg_blur = cv2.GaussianBlur(clean_gray, (5, 5), 0)
    real_texture = cv2.subtract(clean_gray, bg_blur)

    # Threshold to find where the defect is vs the background
    thresh_val = np.mean(dirty_gray) * 0.85 
    _, binary_shape = cv2.threshold(dirty_gray, thresh_val, 255, cv2.THRESH_BINARY)
    soft_shape = cv2.GaussianBlur(binary_shape, (3, 3), 0)
    
    # Determine local contrast
    mask_lines = clean_gray > thresh_val
    mask_gaps = clean_gray <= thresh_val
    val_line = np.mean(clean_gray[mask_lines]) if np.any(mask_lines) else 160
    val_gap = np.mean(clean_gray[mask_gaps]) if np.any(mask_gaps) else 40
    
    norm_shape = soft_shape.astype(float) / 255.0
    base_layer = (val_line * norm_shape + val_gap * (1 - norm_shape)).astype(np.uint8)
    
    # Graft: Base Layer + Real Grain
    final_result = cv2.add(base_layer, real_texture)
    
    return cv2.merge([final_result, final_result, final_result])

# ================= BACKGROUND GENERATION LOGIC =================

def rotate_tile_safe(mat, angle):
    """Rotates a tile without cutting off corners (uses border replication)."""
    if angle == 0: return mat
    h, w = mat.shape[:2]
    big_img = cv2.copyMakeBorder(mat, 2*h, 2*h, 2*w, 2*w, cv2.BORDER_REPLICATE)
    bh, bw = big_img.shape[:2]
    center = (bw // 2, bh // 2)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box
    abs_cos, abs_sin = abs(rot_mat[0,0]), abs(rot_mat[0,1])
    nw = int(bh * abs_sin + bw * abs_cos)
    nh = int(bh * abs_cos + bw * abs_sin)
    
    rot_mat[0, 2] += nw/2 - center[0]
    rot_mat[1, 2] += nh/2 - center[1]
    
    rotated = cv2.warpAffine(big_img, rot_mat, (nw, nh), borderMode=cv2.BORDER_REPLICATE)
    
    # Crop back to center
    rh, rw = rotated.shape[:2]
    sy, sx = (rh // 2) - (h // 2), (rw // 2) - (w // 2)
    return rotated[sy:sy+h, sx:sx+w]

def generate_backgrounds_if_needed():
    """Generates clean background tiles if they don't exist yet."""
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

        crops_done = 0
        attempts = 0
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

# ================= COORDINATE MATH (ROTATION FIX) =================

def get_rotated_single_line_coords(img, angle_deg):
    """
    Finds a single line and a point on it, accounting for rotation.
    Returns: center_point (x,y), line_width, gap_width, unit_vectors
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # 1. Un-rotate to vertical
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    straight_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    if len(straight_img.shape) == 3: straight_gray = cv2.cvtColor(straight_img, cv2.COLOR_BGR2GRAY)
    else: straight_gray = straight_img

    # 2. Find Peaks (Lines) and Valleys (Gaps)
    strip_h = 40
    center_strip = straight_gray[cy-strip_h//2 : cy+strip_h//2, :]
    profile = np.mean(center_strip, axis=0)
    
    peaks, _ = find_peaks(profile, height=np.mean(profile), distance=15, prominence=5)
    valleys, _ = find_peaks(-profile, distance=15, prominence=5)
    
    if len(peaks) < 1: return None

    # 3. Pick a target line
    line_x = random.choice(peaks)
    
    # Estimate widths
    # Find closest valleys to this peak
    left_v = valleys[valleys < line_x]
    right_v = valleys[valleys > line_x]
    
    if len(left_v) == 0 or len(right_v) == 0: return None
    
    v_l = left_v[-1]
    v_r = right_v[0]
    line_width = v_r - v_l
    gap_width = (v_r - v_l) * 0.8 # approx
    
    # Pick Y coord (randomly along the line)
    line_y = random.randint(50, h - 50)
    
    # 4. Rotation Vectors
    # In straight frame: Line is (0,1), Gap is (1,0)
    # We need to rotate these vectors by +angle
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # u_line: Vector pointing ALONG the line
    u_line = np.array([-s, c]) 
    # u_gap: Vector pointing ACROSS the line (Perpendicular)
    u_gap = np.array([c, s])
    
    # Rotate the center point
    pt_straight = np.array([line_x, line_y, 1])
    M_inv = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    pt_rot = M_inv.dot(pt_straight)
    center = (int(pt_rot[0]), int(pt_rot[1]))
    
    return center, int(line_width), int(gap_width), u_line, u_gap

# ================= SHAPE GENERATORS =================

def draw_torn_open(mask, center, line_width, u_line, u_gap):
    """Creates a full line break with jagged, torn edges."""
    break_size = int(line_width * random.uniform(0.3, 0.7))
    half_w = line_width / 2.0
    half_h = break_size / 2.0
    
    corners = [
        center - (u_gap * half_w) - (u_line * half_h), # TL
        center + (u_gap * half_w) - (u_line * half_h), # TR
        center + (u_gap * half_w) + (u_line * half_h), # BR
        center - (u_gap * half_w) + (u_line * half_h)  # BL
    ]
    
    jagged_poly = []
    for pt in corners:
        jitter_x = random.randint(-2, 2)
        jitter_y = random.randint(-2, 2)
        jagged_poly.append((int(pt[0] + jitter_x), int(pt[1] + jitter_y)))
        
    cv2.fillPoly(mask, [np.array(jagged_poly, np.int32)], 255)

def draw_mousebite(mask, center, line_width, u_line, u_gap):
    """Bites a chunk out of the SIDE of the line."""
    side = 1 if random.random() < 0.5 else -1
    edge_center = center + (u_gap * (line_width/2 * side))
    radius = int(line_width * random.uniform(0.3, 0.5))
    cx, cy = int(edge_center[0]), int(edge_center[1])
    pts = []
    num_verts = random.randint(8, 12)
    for i in range(num_verts):
        angle = (i / num_verts) * 2 * np.pi
        r_var = radius * random.uniform(0.8, 1.2)
        px = int(cx + r_var * np.cos(angle))
        py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

def draw_scratch(mask, center, line_width, u_line, u_gap):
    """Thin, jagged scratch cutting across the line."""
    length = int(line_width * 1.5)
    thickness = random.randint(1, 3)
    p1 = center - (u_gap * (length/2))
    p2 = center + (u_gap * (length/2))
    steps = 10
    points = []
    for i in range(steps + 1):
        t = i / steps
        curr = p1 * (1-t) + p2 * t
        jitter = u_line * random.uniform(-2, 2)
        pt = curr + jitter
        points.append((int(pt[0]), int(pt[1])))
    for i in range(len(points)-1):
        cv2.line(mask, points[i], points[i+1], 255, thickness)

# ================= MAIN PIPELINE =================

def generate_opens():
    print("--- Generating OPENS (Rotation Fixed + Texture Grafting) ---")
    bg_files = [f for f in os.listdir(BASE_IMGS_FOLDER) if f.startswith("bg_")]
    if not bg_files: return

    count = 0
    while count < TARGET_COUNT:
        filename = random.choice(bg_files)
        try: 
            src_id = filename.split('_')[2].replace('src', '')
            angle_str = filename.split('_')[-1].replace('.png','').replace('deg','')
            angle = int(angle_str)
        except: continue

        canvas = cv2.imread(os.path.join(BASE_IMGS_FOLDER, filename))
        if canvas is None: continue
        clean_bg_copy = canvas.copy()
        
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        
        # Gap Statistics (Darkest 20%)
        darkest_pixels = gray[gray < np.percentile(gray, 20)]
        if len(darkest_pixels) > 0:
            gap_mean = np.mean(darkest_pixels)
            gap_std = np.std(darkest_pixels)
        else:
            gap_mean, gap_std = 30, 5

        # Rotation-Aware Detection
        res = get_rotated_single_line_coords(canvas, angle)
        if res is None: continue
        center, line_width, gap_width, u_line, u_gap = res
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        open_type = random.random()
        
        if open_type < 0.60:
            draw_torn_open(mask, center, line_width, u_line, u_gap) 
        elif open_type < 0.85:
            draw_mousebite(mask, center, line_width, u_line, u_gap) 
        else:
            draw_scratch(mask, center, line_width, u_line, u_gap) 

        if np.sum(mask) == 0: continue

        mask = roughen_mask_edges(mask)
        tex = get_mapped_texture(src_id, '_b', 224, 224, target_val_stats=(gap_mean, gap_std))
        if angle == 90: tex = cv2.rotate(tex, cv2.ROTATE_90_CLOCKWISE)
        tex = cv2.resize(tex, (224, 224))
        
        mask_soft = cv2.GaussianBlur(mask, (3, 3), 0)
        alpha = cv2.cvtColor(mask_soft, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
        dirty_canvas = (canvas.astype(float) * (1 - alpha) + tex.astype(float) * alpha).astype(np.uint8)
        
        final_img_pre_noise = apply_texture_grafting(clean_bg_copy, dirty_canvas)
        final_img = apply_global_grit(final_img_pre_noise)
        
        out_name = f"open_{count:06d}.png"
        cv2.imwrite(f"{FINAL_OUTPUT_DIR}/images/{out_name}", final_img)
        
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