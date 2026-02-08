import cv2
import numpy as np
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER  = 'Via_Arrays_Base_Imgs'
OUTPUT_ROOT   = 'Staging_Dataset/Via Array Defects' # <--- UPDATED PATH
TARGET_COUNT  = 1000  
TARGET_SIZE   = (224, 224)

# Class IDs
CLASS_ID_MISSING_VIA = 0
CLASS_ID_PARTICLE    = 1
CLASS_ID_CMP         = 2

# Output Directories
IMG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'images')
LBL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'labels')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

# ==========================================
# UTILS: OVERLAP DETECTION
# ==========================================
def check_overlap(new_cx, new_cy, new_r, existing_defects, buffer=0.05):
    """
    Checks if a new circular region overlaps with any existing defect bounding boxes.
    All inputs should be normalized (0-1).
    """
    # Convert circle to approx box [x_min, y_min, x_max, y_max]
    n_x1 = new_cx - new_r
    n_x2 = new_cx + new_r
    n_y1 = new_cy - new_r
    n_y2 = new_cy + new_r

    for defect in existing_defects:
        # defect format: (class_id, cx, cy, w, h)
        _, d_cx, d_cy, d_w, d_h = defect
        
        d_x1 = d_cx - d_w/2 - buffer
        d_x2 = d_cx + d_w/2 + buffer
        d_y1 = d_cy - d_h/2 - buffer
        d_y2 = d_cy + d_h/2 + buffer

        # Check for intersection
        if (n_x1 < d_x2 and n_x2 > d_x1 and
            n_y1 < d_y2 and n_y2 > d_y1):
            return True # Overlap detected
            
    return False

# ==========================================
# 1. MISSING VIA LOGIC
# ==========================================
def setup_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 220
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.60
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    return cv2.SimpleBlobDetector_create(params)

BLOB_DETECTOR = setup_blob_detector()

def apply_missing_via(img, existing_defects):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    keypoints = BLOB_DETECTOR.detect(enhanced_gray)

    if not keypoints:
        return img, None

    # Filter valid keypoints (those NOT overlapping existing defects)
    valid_kps = []
    for kp in keypoints:
        cx, cy = kp.pt
        r = kp.size / 2
        
        norm_cx, norm_cy = cx/w, cy/h
        norm_r = r/w # approx
        
        if not check_overlap(norm_cx, norm_cy, norm_r, existing_defects):
            valid_kps.append(kp)
            
    if not valid_kps:
        return img, None

    # Pick 1 valid via
    target_kp = random.choice(valid_kps)
    cx, cy = int(target_kp.pt[0]), int(target_kp.pt[1])
    r = int(target_kp.size / 2)
    
    # Inpainting logic
    defect_r = int(r * 1.6)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), defect_r, 255, -1)
    
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    # Add Noise
    noise = np.zeros(img.shape, dtype=np.int16)
    cv2.randn(noise, (0,0,0), (15, 15, 15))
    noise_smooth = cv2.GaussianBlur(noise, (3, 3), 0)
    img_int = img.astype(np.int16)
    img_noisy = cv2.add(img_int, noise_smooth)
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    mask_bool = mask > 0
    img[mask_bool] = img_noisy[mask_bool]

    # Edge Blur
    mask_edge = cv2.Canny(mask, 100, 200)
    mask_edge = cv2.dilate(mask_edge, np.ones((3,3), np.uint8))
    img_blurred = cv2.GaussianBlur(img, (3,3), 0)
    img[mask_edge > 0] = img_blurred[mask_edge > 0]

    norm_x = cx / w
    norm_y = cy / h
    norm_w = (r * 2) / w
    norm_h = (r * 2) / h
    
    return img, (CLASS_ID_MISSING_VIA, norm_x, norm_y, norm_w, norm_h)

# ==========================================
# 2. PARTICLE LOGIC
# ==========================================
def draw_organic_blob(mask, center, radius, jaggedness=0.2):
    cx, cy = center
    pts = []
    num_pts = random.randint(8, 14) 
    for i in range(num_pts):
        angle = (i / num_pts) * 2 * np.pi
        r_var = radius * random.uniform(1.0 - jaggedness, 1.0 + jaggedness)
        px = int(cx + r_var * np.cos(angle))
        py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

def apply_sem_particle_style(mask, is_bright_type):
    h, w = mask.shape
    tex = np.zeros((h, w), dtype=np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0: dist = dist / dist.max()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    rim_mask = cv2.subtract(dilated, mask)

    if is_bright_type:
        core_val = 200 + (dist * 55)
        noise = np.random.normal(0, 15, (h, w))
        core_val = np.clip(core_val + noise, 180, 255)
        tex[mask > 0] = core_val[mask > 0].astype(np.uint8)
        tex[rim_mask > 0] = 30 
    else:
        core_val = 40 + (dist * 20)
        noise = np.random.normal(0, 5, (h, w))
        core_val = np.clip(core_val + noise, 20, 80)
        tex[mask > 0] = core_val[mask > 0].astype(np.uint8)
        tex[rim_mask > 0] = 180 

    return tex, dilated

def apply_particle(img, existing_defects):
    h, w = img.shape[:2]
    
    # Retry loop to find non-overlapping spot
    for attempt in range(30):
        # Generate random spot
        cx, cy = random.randint(20, w-20), random.randint(20, h-20)
        # Estimate max size (approx 20px radius safe estimate)
        norm_cx, norm_cy = cx/w, cy/h
        norm_r = 25.0 / w 

        if not check_overlap(norm_cx, norm_cy, norm_r, existing_defects):
            # Safe to draw
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Single or Cluster
            if random.random() < 0.65:
                size = random.randint(6, 15)
                draw_organic_blob(mask, (cx, cy), size, jaggedness=0.3)
            else:
                num_sub = random.randint(3, 7)
                for _ in range(num_sub):
                    ox, oy = random.randint(-10, 10), random.randint(-10, 10)
                    draw_organic_blob(mask, (cx+ox, cy+oy), random.randint(3, 8), jaggedness=0.2)
            
            if np.sum(mask) == 0: continue

            is_charging = random.random() < 0.90
            particle_tex, blend_mask = apply_sem_particle_style(mask, is_charging)
            
            mask_soft = cv2.GaussianBlur(blend_mask, (3, 3), 0)
            alpha = mask_soft.astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            
            part_bgr = cv2.cvtColor(particle_tex, cv2.COLOR_GRAY2BGR)
            img = (img.astype(float) * (1 - alpha) + part_bgr.astype(float) * alpha).astype(np.uint8)

            # Calc Label
            contours, _ = cv2.findContours(blend_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return img, None
            c = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(c)
            norm_x, norm_y = (x + bw/2)/w, (y + bh/2)/h
            norm_w, norm_h = bw/w, bh/h
            
            return img, (CLASS_ID_PARTICLE, norm_x, norm_y, norm_w, norm_h)
            
    return img, None

# ==========================================
# 3. CMP RESIDUE LOGIC
# ==========================================
def generate_micro_cluster_shape(h, w, cx, cy):
    mask = np.zeros((h, w), dtype=np.float32)
    num_blobs = random.randint(3, 10)
    stretch_angle = random.uniform(0, np.pi)
    stretch_factor = random.uniform(1.2, 2.5) 
    
    for _ in range(num_blobs):
        offset_r = random.uniform(0, 14) 
        offset_theta = random.uniform(0, 2*np.pi)
        bx = int(cx + offset_r * np.cos(offset_theta))
        by = int(cy + offset_r * np.sin(offset_theta))
        
        axis_len_small = random.randint(6, 14)
        axis_len_large = int(axis_len_small * stretch_factor)
        
        cv2.ellipse(mask, (bx, by), (axis_len_large, axis_len_small), 
                    np.degrees(stretch_angle) + random.uniform(-20, 20), 0, 360, 1.0, -1)
        
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    mask = np.where(mask > 0.3, 1.0, 0.0).astype(np.float32)
    
    noise = np.zeros((h, w), dtype=np.float32)
    cv2.randu(noise, 0, 1.0)
    
    if random.random() < 0.5:
        noise = cv2.GaussianBlur(noise, (7, 7), 0)
        final_mask = np.where((mask - (noise * 0.2)) > 0.5, 1.0, 0.0)
    else:
        final_mask = np.where((mask - (noise * 0.4)) > 0.5, 1.0, 0.0)
        
    return final_mask.astype(np.float32), stretch_angle

def add_micro_satellites(mask, cx, cy):
    h, w = mask.shape
    for _ in range(random.randint(0, 6)):
        r = random.uniform(15, 35)
        theta = random.uniform(0, 2*np.pi)
        sx = int(cx + r * np.cos(theta))
        sy = int(cy + r * np.sin(theta))
        if 0 <= sx < w and 0 <= sy < h:
            cv2.circle(mask, (sx, sy), random.randint(1, 2), 1.0, -1)
    return mask

def apply_cmp_residue(img, existing_defects):
    h, w = img.shape[:2]
    
    # Retry loop
    for attempt in range(30):
        cx, cy = random.randint(40, w-40), random.randint(40, h-40)
        
        # Estimate Max Radius for overlap check (approx 45px)
        norm_cx, norm_cy = cx/w, cy/h
        norm_r = 45.0 / w
        
        if not check_overlap(norm_cx, norm_cy, norm_r, existing_defects):
            # Safe to draw
            canvas = img.astype(np.float32)
            bg_mean = np.mean(canvas)
            
            mask, motion_angle = generate_micro_cluster_shape(h, w, cx, cy)
            mask = add_micro_satellites(mask, cx, cy)
            
            if np.sum(mask) == 0: continue

            # Erosion
            erosion_zone = cv2.GaussianBlur(mask, (19, 19), 0)
            erosion_strength = 0.95 if random.random() < 0.05 else 0.7
            for c in range(3):
                canvas[:,:,c] = canvas[:,:,c] * (1 - erosion_zone * erosion_strength) + (bg_mean * erosion_zone * erosion_strength)

            # Texture
            Y, X = np.ogrid[:h, :w]
            dist_along_axis = (X - cx) * np.cos(motion_angle) + (Y - cy) * np.sin(motion_angle)
            
            mask_indices = mask > 0
            intensity_grad = 0
            if np.any(mask_indices):
                g_min, g_max = dist_along_axis[mask_indices].min(), dist_along_axis[mask_indices].max()
                norm_grad = (dist_along_axis - g_min) / (g_max - g_min + 1e-5)
                intensity_grad = (norm_grad - 0.5) * random.uniform(-30, -10)
                
            patch_base = bg_mean + random.uniform(30, 60)
            grain = np.zeros((h, w), dtype=np.float32)
            cv2.randu(grain, -10, 10)
            patch_tex = patch_base + grain + intensity_grad

            # Shading
            height_map = cv2.GaussianBlur(mask, (7, 7), 0) 
            gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
            shading = (gx + gy) * 40
            patch_tex += shading

            # Paste
            mask_3d = mask[:, :, np.newaxis]
            patch_3d = np.repeat(patch_tex[:, :, np.newaxis], 3, axis=2)
            
            shadow = cv2.GaussianBlur(mask, (7, 7), 0) * 0.5
            canvas *= (1 - shadow[:, :, np.newaxis])
            canvas = (canvas * (1 - mask_3d)) + (patch_3d * mask_3d)
            
            final_img = np.clip(canvas, 0, 255).astype(np.uint8)

            # Label
            coords = np.argwhere(mask > 0)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [CLASS_ID_CMP, cx/w, cy/h, (x_max - x_min + 8)/w, (y_max - y_min + 8)/h]
            
            return final_img, bbox
            
    return img, None

# ==========================================
# MASTER FACTORY
# ==========================================
def process_factory():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: {INPUT_FOLDER} folder missing.")
        return

    files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    if not files: 
        print("No background images found!")
        return

    print(f"Generating {TARGET_COUNT} images with EXACTLY 2 NON-OVERLAPPING DEFECTS...")

    for i in range(TARGET_COUNT):
        img_choice = random.choice(files)
        img = cv2.imread(img_choice)
        img = cv2.resize(img, TARGET_SIZE)
        
        existing_defects = [] # Stores (class_id, cx, cy, w, h)
        
        # We need exactly 2 defects
        defects_to_place = 2
        
        while defects_to_place > 0:
            # Randomly pick a defect type
            # 0: Missing Via, 1: Particle, 2: CMP
            d_type = random.choice([0, 1, 2])
            
            new_img = None
            new_lbl = None
            
            if d_type == 0:
                new_img, new_lbl = apply_missing_via(img, existing_defects)
            elif d_type == 1:
                new_img, new_lbl = apply_particle(img, existing_defects)
            else:
                new_img, new_lbl = apply_cmp_residue(img, existing_defects)
            
            # If successful placement
            if new_lbl is not None:
                img = new_img
                existing_defects.append(new_lbl)
                defects_to_place -= 1
            else:
                # Failed to place (probably due to overlap), loop will retry
                pass

        # Save Result
        name = f'DUAL_DEFECT_{i:04d}.png'
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, name), img)
        
        with open(os.path.join(LBL_OUTPUT_DIR, name.replace('.png', '.txt')), 'w') as f:
            for b in existing_defects:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        if i % 100 == 0:
            print(f"Processed {i}/{TARGET_COUNT}...")

    print(f"\nSuccess! Generated {TARGET_COUNT} dual-defect images in {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_factory()