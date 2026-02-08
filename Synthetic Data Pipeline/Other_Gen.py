import cv2
import numpy as np
import random
import os

# ================= CONFIGURATION =================
SRC_CLEAN_BGS = 'Staging_Dataset/Clean/images'
OUTPUT_ROOT   = 'Staging_Dataset'

# TOTAL TARGET (Will be split evenly 1/3 per type)
TARGET_PARTICLE_TOTAL = 3000  
CLASS_ID_PARTICLE = 6 

os.makedirs(f'{OUTPUT_ROOT}/Particle/images', exist_ok=True)
os.makedirs(f'{OUTPUT_ROOT}/Particle/labels', exist_ok=True)

# ================= UTILS =================

def apply_sem_particle_style(mask, is_bright_type):
    """
    Apply advanced SEM physics to distinguish particles from bridges.
    """
    h, w = mask.shape
    tex = np.zeros((h, w), dtype=np.uint8)
    
    # Distance transform creates the "Dome" shape
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    
    # Create the Effect Zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    rim_mask = cv2.subtract(dilated, mask) 
    
    if is_bright_type:
        # === CHARGING DUST (White) ===
        core_val = 200 + (dist * 55)
        noise = np.random.normal(0, 15, (h, w))
        core_val = np.clip(core_val + noise, 180, 255)
        tex[mask > 0] = core_val[mask > 0].astype(np.uint8)
        tex[rim_mask > 0] = 30 # Dark halo
    else:
        # === ORGANIC / CARBON (Dark) ===
        core_val = 40 + (dist * 20)
        noise = np.random.normal(0, 5, (h, w))
        core_val = np.clip(core_val + noise, 20, 80)
        tex[mask > 0] = core_val[mask > 0].astype(np.uint8)
        tex[rim_mask > 0] = 180 # Bright rim

    return tex, dilated

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

def save_labels(mask, filepath, class_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    with open(filepath, 'w') as f:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w < 3 or h < 3: continue 
            cx, cy = (x + w/2)/224, (y + h/2)/224
            nw, nh = w/224, h/224
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# ================= MAIN GENERATOR =================
def generate_particles():
    print(f"--- Generating {TARGET_PARTICLE_TOTAL} PARTICLES (Balanced Backgrounds) ---")
    
    if not os.path.exists(SRC_CLEAN_BGS):
        print(f"Error: {SRC_CLEAN_BGS} not found.")
        return

    # 1. SORT FILES BY TYPE
    all_files = [f for f in os.listdir(SRC_CLEAN_BGS) if f.lower().endswith(('.png', '.jpg'))]
    
    files_striped = [f for f in all_files if 'metal_ls' in f]
    files_via     = [f for f in all_files if 'via_array' in f]
    files_cmp     = [f for f in all_files if 'dummy_cmp' in f]
    
    print(f"   Source Stats: Metal={len(files_striped)} | Via={len(files_via)} | CMP={len(files_cmp)}")
    
    if not (files_striped and files_via and files_cmp):
        print("Warning: One or more background types missing. Falling back to random mix.")
        bg_queue = all_files # Fallback
    else:
        # Create a perfectly balanced queue
        # 1000 of each type
        target_per_type = TARGET_PARTICLE_TOTAL // 3
        
        # Sample with replacement if we don't have enough source images
        q_striped = random.choices(files_striped, k=target_per_type)
        q_via     = random.choices(files_via, k=target_per_type)
        q_cmp     = random.choices(files_cmp, k=target_per_type)
        
        bg_queue = q_striped + q_via + q_cmp
        random.shuffle(bg_queue) # Randomize order so we don't generate all stripes first

    count = 0
    # Limit to target total in case of rounding errors
    for filename in bg_queue[:TARGET_PARTICLE_TOTAL]:
        
        canvas = cv2.imread(os.path.join(SRC_CLEAN_BGS, filename))
        if canvas is None: continue
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        
        # Cluster vs Single
        mode = random.random()
        
        if mode < 0.65:
            # === SINGLE PARTICLE (Size 6-15px) ===
            cx, cy = random.randint(20, 204), random.randint(20, 204)
            size = random.randint(6, 15) 
            draw_organic_blob(mask, (cx, cy), size, jaggedness=0.3)
        else:
            # === CLUSTER (Grape Bunch) ===
            clust_cx, clust_cy = random.randint(25, 199), random.randint(25, 199)
            num_sub = random.randint(3, 7)
            for _ in range(num_sub):
                offset_x = random.randint(-10, 10)
                offset_y = random.randint(-10, 10)
                sub_size = random.randint(3, 8)
                draw_organic_blob(mask, (clust_cx+offset_x, clust_cy+offset_y), sub_size, jaggedness=0.2)
        
        if np.sum(mask) == 0: continue
        
        # Physics: 90% Charging (Bright), 10% Dark
        is_charging = random.random() < 0.90
        
        # Apply the SEM Physics style
        particle_tex, blend_mask = apply_sem_particle_style(mask, is_charging)
        
        # Blend Logic
        mask_soft = cv2.GaussianBlur(blend_mask, (3, 3), 0)
        alpha = mask_soft.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        part_bgr = cv2.cvtColor(particle_tex, cv2.COLOR_GRAY2BGR)
        
        # Standard Alpha Blend
        final_img = (canvas.astype(float) * (1 - alpha) + part_bgr.astype(float) * alpha).astype(np.uint8)

        # Output
        out_name = f"part_{count:06d}.png"
        cv2.imwrite(f"{OUTPUT_ROOT}/Particle/images/{out_name}", final_img)
        save_labels(mask, f"{OUTPUT_ROOT}/Particle/labels/{out_name.replace('.png','.txt')}", CLASS_ID_PARTICLE)
        
        count += 1
        if count % 500 == 0: print(f"Particles: {count}/{TARGET_PARTICLE_TOTAL}")

if __name__ == "__main__":
    generate_particles()