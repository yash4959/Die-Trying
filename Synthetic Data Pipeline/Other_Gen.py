import cv2
import numpy as np
import random
import os
import math

# ==========================================
# CONFIGURATION
# ==========================================
SRC_CLEAN_BGS = 'Staging_Dataset/No_Defect/images'
OUTPUT_ROOT   = 'Staging_Dataset'

# Targets
TARGET_PARTICLE = 4000  # Total particles

# Class ID for YOLO (6 = Particle)
CLASS_ID_PARTICLE = 6 

# Setup
os.makedirs(f'{OUTPUT_ROOT}/Particle/images', exist_ok=True)
os.makedirs(f'{OUTPUT_ROOT}/Particle/labels', exist_ok=True)

# ==========================================
# UTILS
# ==========================================
def draw_organic_blob(mask, center, radius, jaggedness=0.2):
    cx, cy = center
    pts = []
    num_pts = random.randint(6, 12) 
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
            if w < 2 or h < 2: continue 
            cx, cy = (x + w/2)/224, (y + h/2)/224
            nw, nh = w/224, h/224
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

# ==========================================
# MAIN GENERATOR
# ==========================================
def generate_particles():
    print(f"--- Generating {TARGET_PARTICLE} PARTICLES (Balanced Mix) ---")
    
    if not os.path.exists(SRC_CLEAN_BGS): 
        print(f"Error: {SRC_CLEAN_BGS} folder missing!"); return
    
    # 1. SORT FILES BY TYPE
    all_files = [f for f in os.listdir(SRC_CLEAN_BGS) if f.lower().endswith(('.png', '.jpg'))]
    
    files_striped = [f for f in all_files if 'bg_striped' in f]
    files_cmp     = [f for f in all_files if 'bg_cmp' in f]
    files_via     = [f for f in all_files if 'bg_via' in f]
    
    print(f"  Source Distribution: Striped={len(files_striped)}, CMP={len(files_cmp)}, Via={len(files_via)}")
    
    if not (files_striped and files_cmp and files_via):
        print("Warning: One or more background types are missing! Falling back to random shuffle.")
        # Fallback if names don't match expected pattern
        selected_files = random.sample(all_files, TARGET_PARTICLE)
    else:
        # 2. FORCE EQUAL SPLIT
        quota = TARGET_PARTICLE // 3
        remainder = TARGET_PARTICLE % 3
        
        # Helper to safely sample with replacement if needed
        def get_sample(source_list, k):
            if len(source_list) < k:
                return (source_list * (k // len(source_list) + 1))[:k]
            return random.sample(source_list, k)

        print(f"  Target per type: ~{quota}")
        
        batch_striped = get_sample(files_striped, quota + remainder) # Give remainder to striped
        batch_cmp     = get_sample(files_cmp, quota)
        batch_via     = get_sample(files_via, quota)
        
        selected_files = batch_striped + batch_cmp + batch_via
        random.shuffle(selected_files) # Mix them up for processing

    count = 0
    for filename in selected_files:
        canvas = cv2.imread(os.path.join(SRC_CLEAN_BGS, filename))
        if canvas is None: continue
        
        mask = np.zeros((224, 224), dtype=np.uint8)
        
        # 1 to 5 particles per image
        num_particles = random.randint(1, 5)
        
        for _ in range(num_particles):
            cx = random.randint(5, 219)
            cy = random.randint(5, 219)
            
            # Small Dust Size (Radius 1-5px)
            size = random.randint(1, 5) 
            
            ptype = random.random()
            if ptype < 0.5: 
                draw_organic_blob(mask, (cx, cy), size, jaggedness=0.5)
            elif ptype < 0.8: 
                axes = (size, int(size * random.uniform(0.3, 0.7)))
                angle = random.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
            else: 
                cv2.circle(mask, (cx, cy), size, 255, -1)
            
        if np.sum(mask) == 0: continue
        
        # Physics: 70% Charging (Bright), 30% Carbon (Dark)
        is_charging = random.random() < 0.70
        if is_charging:
            intensity = random.randint(200, 255)
        else:
            intensity = random.randint(30, 80)
            
        noise = np.random.normal(0, 10, mask.shape).astype(np.int16)
        particle_tex = np.full(mask.shape, intensity, dtype=np.int16) + noise
        particle_tex = np.clip(particle_tex, 0, 255).astype(np.uint8)
        
        mask_soft = cv2.GaussianBlur(mask, (3, 3), 0)
        alpha = mask_soft.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        part_bgr = cv2.cvtColor(particle_tex, cv2.COLOR_GRAY2BGR)
        final_img = (canvas.astype(float) * (1 - alpha) + part_bgr.astype(float) * alpha).astype(np.uint8)

        out_name = f"part_{count:06d}.png"
        cv2.imwrite(f"{OUTPUT_ROOT}/Particle/images/{out_name}", final_img)
        save_labels(mask, f"{OUTPUT_ROOT}/Particle/labels/{out_name.replace('.png','.txt')}", CLASS_ID_PARTICLE)
        
        count += 1
        if count % 500 == 0: print(f"Particles: {count}/{TARGET_PARTICLE}")

if __name__ == "__main__":
    generate_particles()