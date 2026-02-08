import cv2
import numpy as np
import random
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
OUTPUT_ROOT = 'Staging_Dataset'

# --- SOURCES ---
SAFE_FOLDER_CMP    = 'Dummy_Fill_Raw_Imgs'   # Background Source
CMP_INJECT_SOURCE  = 'CMP_Mined_Patches'
CMP_DIRECT_SOURCE  = 'CMP_Manual_Patches'
CRACK_SOURCE       = 'Crack_Raw_Imgs'

# --- TARGETS ---
TARGET_MIXED_IMAGES = 1000  
CMP_CLASS_ID       = 3
CRACK_CLASS_ID     = 4
PARTICLE_CLASS_ID  = 6

# --- SETTINGS ---
TARGET_SIZE        = (224, 224)
CANVAS_SIZE        = (800, 800)
TILES_TO_SPLATTER  = 400 

# Create Output Directories
os.makedirs(f'{OUTPUT_ROOT}/Mixed_Defects/images', exist_ok=True)
os.makedirs(f'{OUTPUT_ROOT}/Mixed_Defects/labels', exist_ok=True)

# ==========================================
# 2. RESOURCE LOADING
# ==========================================
print("--- Loading Resources ---")

# A. Background Tiles
safe_tiles = []
if os.path.exists(SAFE_FOLDER_CMP):
    for f in os.listdir(SAFE_FOLDER_CMP):
        img = cv2.imread(os.path.join(SAFE_FOLDER_CMP, f))
        if img is not None: safe_tiles.append(img)

# B. CMP Patches
cmp_pool = []
for src in [CMP_DIRECT_SOURCE, CMP_INJECT_SOURCE]:
    if os.path.exists(src):
        for f in os.listdir(src):
            img = cv2.imread(os.path.join(src, f), cv2.IMREAD_UNCHANGED)
            if img is not None:
                # Ensure Alpha for CMP blending
                if img.shape[2] == 3 and src == CMP_INJECT_SOURCE:
                    b, g, r = cv2.split(img)
                    alpha = np.ones_like(b) * 255
                    img = cv2.merge([b, g, r, alpha])
                cmp_pool.append(img)

# C. Crack Patches
crack_pool = []
if os.path.exists(CRACK_SOURCE):
    for f in os.listdir(CRACK_SOURCE):
        img = cv2.imread(os.path.join(CRACK_SOURCE, f), cv2.IMREAD_UNCHANGED)
        if img is not None: crack_pool.append(img)

if not safe_tiles:
    print("CRITICAL ERROR: No background tiles found!")
    exit()

print(f"Loaded: {len(safe_tiles)} BG Tiles | {len(cmp_pool)} CMP | {len(crack_pool)} Cracks")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    border_val = (255, 255, 255, 0) if image.shape[2] == 4 else (255, 255, 255)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)

# --- CMP BLENDER ---
def blend_cmp_natural(base_roi, patch):
    ph, pw = patch.shape[:2]
    rh, rw = base_roi.shape[:2]
    if (ph != rh) or (pw != rw): patch = cv2.resize(patch, (rw, rh))
    
    if patch.shape[2] == 4:
        b, g, r, a = cv2.split(patch)
        patch_rgb = cv2.merge([b, g, r])
        if np.min(a) < 250: mask = a
        else: # Auto-mask
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

# --- CRACK BLENDER ---
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

# --- PARTICLE GENERATOR (MISSING FUNCTIONS ADDED HERE) ---

def draw_organic_blob(mask, center, radius, jaggedness=0.2):
    """Draws a random organic shape for particles."""
    cx, cy = center
    pts = []
    num_pts = random.randint(8, 14) 
    for i in range(num_pts):
        angle = (i / num_pts) * 2 * np.pi
        r_var = radius * random.uniform(1.0 - jaggedness, 1.0 + jaggedness)
        px = int(cx + r_var * np.cos(angle))
        py = int(cy + r_var * np.sin(angle))
        pts.append((px, py))
    # Draw solid white blob on mask
    cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)

def apply_sem_particle_style(mask):
    """Converts a flat mask into a realistic SEM particle texture."""
    h, w = mask.shape
    tex = np.zeros((h, w), dtype=np.uint8)
    
    # Distance transform for 3D dome effect
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0: dist = dist / dist.max()
    
    # Create the Halo Zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    rim_mask = cv2.subtract(dilated, mask) 
    
    # Charging Style (Bright White Center)
    core_val = 200 + (dist * 55)
    noise = np.random.normal(0, 15, (h, w))
    core_val = np.clip(core_val + noise, 180, 255)
    
    tex[mask > 0] = core_val[mask > 0].astype(np.uint8)
    tex[rim_mask > 0] = 30 # Dark halo ring
    
    return tex, dilated

# ==========================================
# 4. MASTER GENERATOR
# ==========================================

def generate_mixed_defects():
    print(f"\n--- Generating {TARGET_MIXED_IMAGES} Mixed Defect Images ---")
    
    success_count = 0
    while success_count < TARGET_MIXED_IMAGES:
        
        # 1. CREATE CANVAS (Mosaic Background)
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

        # 2. DEFECT LOGIC
        has_cmp      = True  
        has_crack    = random.random() < 0.70
        has_particle = random.random() < 0.60

        global_bboxes = [] # (class_id, x, y, w, h)
        
        # Hot Zone
        zone_x = random.randint(200, 500)
        zone_y = random.randint(200, 500)

        # --- LAYER 1: CMP SCRATCHES ---
        if has_cmp and cmp_pool:
            p = random.choice(cmp_pool).copy()
            p = np.rot90(p, random.randint(0, 3))
            s = random.uniform(0.8, 1.0)
            nw, nh = int(p.shape[1] * s), int(p.shape[0] * s)
            p = cv2.resize(p, (nw, nh))
            xs, ys = zone_x + random.randint(-50, 50), zone_y + random.randint(-50, 50)
            roi = canvas[ys:ys+nh, xs:xs+nw].copy()
            canvas[ys:ys+nh, xs:xs+nw] = blend_cmp_natural(roi, p)
            global_bboxes.append((CMP_CLASS_ID, xs, ys, nw, nh))

        # --- LAYER 2: CRACKS ---
        if has_crack and crack_pool:
            p = random.choice(crack_pool).copy()
            p = rotate_image(p, random.randint(0, 360))
            s = random.uniform(0.15, 0.30)
            nw, nh = int(p.shape[1] * s), int(p.shape[0] * s)
            p = cv2.resize(p, (nw, nh))
            xs, ys = zone_x + random.randint(-50, 50), zone_y + random.randint(-50, 50)
            roi = canvas[ys:ys+nh, xs:xs+nw]
            canvas[ys:ys+nh, xs:xs+nw] = blend_crack_visible(roi, p)
            global_bboxes.append((CRACK_CLASS_ID, xs, ys, nw, nh))

        # --- LAYER 3: PARTICLES (On Top) ---
        if has_particle:
            particle_mask = np.zeros(CANVAS_SIZE[:2], dtype=np.uint8)
            px, py = zone_x + random.randint(-40, 40), zone_y + random.randint(-40, 40)
            size = random.randint(6, 15)
            
            # CALLING THE NOW-DEFINED FUNCTION
            draw_organic_blob(particle_mask, (px, py), size, jaggedness=0.3)
            
            # Physics Rendering
            p_tex_full, blend_mask_full = apply_sem_particle_style(particle_mask)
            
            # Blend
            mask_soft = cv2.GaussianBlur(blend_mask_full, (3, 3), 0)
            alpha = mask_soft.astype(float) / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            part_bgr = cv2.cvtColor(p_tex_full, cv2.COLOR_GRAY2BGR)
            canvas = (canvas.astype(float) * (1 - alpha) + part_bgr.astype(float) * alpha).astype(np.uint8)
            
            # Add BBox
            contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                global_bboxes.append((PARTICLE_CLASS_ID, x, y, w, h))

        # 3. CROP FINAL IMAGE
        cx = max(0, min(zone_x + 30 - 112, CANVAS_SIZE[1] - 224))
        cy = max(0, min(zone_y + 30 - 112, CANVAS_SIZE[0] - 224))
        
        final_img = canvas[cy:cy+224, cx:cx+224]

        # 4. FILTER LABELS
        local_bboxes = []
        for (cid, gx, gy, gw, gh) in global_bboxes:
            nx, ny = gx - cx, gy - cy
            x1, y1 = max(0, nx), max(0, ny)
            x2, y2 = min(224, nx + gw), min(224, ny + gh)
            
            area_visible = (x2-x1) * (y2-y1)
            area_orig = gw * gh
            # Must see at least 30% of the defect
            if area_visible > (area_orig * 0.3) and area_visible > 50:
                cx_norm = (x1 + (x2-x1)/2) / 224
                cy_norm = (y1 + (y2-y1)/2) / 224
                w_norm = (x2-x1) / 224
                h_norm = (y2-y1) / 224
                local_bboxes.append(f"{cid} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # QC: Keep only if we have at least 2 defects visible
        if len(local_bboxes) >= 2:
            out_name = f"mixed_{success_count:06d}"
            cv2.imwrite(f'{OUTPUT_ROOT}/Mixed_Defects/images/{out_name}.png', final_img)
            with open(f'{OUTPUT_ROOT}/Mixed_Defects/labels/{out_name}.txt', 'w') as f:
                for line in local_bboxes:
                    f.write(line + "\n")
            
            success_count += 1
            if success_count % 100 == 0: print(f"  Mixed: {success_count}/{TARGET_MIXED_IMAGES}")

if __name__ == "__main__":
    generate_mixed_defects()
    print("Done.")