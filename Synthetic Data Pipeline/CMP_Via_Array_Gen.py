import cv2
import numpy as np
import os
import random

# --- CONFIGURATION ---
INPUT_FOLDER = 'Via_Arrays_Base_Imgs'
OUTPUT_BASE = 'Staging_Dataset'
CATEGORY = 'CMP'
SAMPLES_PER_CLASS = 1000 
TARGET_SIZE = (224, 224)

IMG_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'images')
LBL_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'labels')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

def generate_micro_cluster_shape(h, w, cx, cy):
    """
    Generates a MEDIUM-SMALL, highly varied organic shape by merging 
    random sub-blobs. Creates 'crumb' or 'splatter' shapes.
    """
    mask = np.zeros((h, w), dtype=np.float32)
    
    # 1. Random Complexity: 3 to 10 sub-components (Slightly more complex)
    num_blobs = random.randint(3, 10)
    
    # Random Directional bias for this specific defect
    stretch_angle = random.uniform(0, np.pi)
    stretch_factor = random.uniform(1.2, 2.5) 
    
    for _ in range(num_blobs):
        # INCREASED Offset: Allow blobs to spread slightly more (0 to 14px)
        offset_r = random.uniform(0, 14) 
        offset_theta = random.uniform(0, 2*np.pi)
        bx = int(cx + offset_r * np.cos(offset_theta))
        by = int(cy + offset_r * np.sin(offset_theta))
        
        # INCREASED Size: 6 to 14 pixels radius equivalent (was 3-9)
        axis_len_small = random.randint(6, 14)
        axis_len_large = int(axis_len_small * stretch_factor)
        
        # Draw ellipse
        cv2.ellipse(mask, 
                    (bx, by), 
                    (axis_len_large, axis_len_small), 
                    np.degrees(stretch_angle) + random.uniform(-20, 20), 
                    0, 360, 1.0, -1)
        
    # 2. Merge blobs
    mask = cv2.GaussianBlur(mask, (11, 11), 0) # Slightly larger blur for larger shapes
    mask = np.where(mask > 0.3, 1.0, 0.0).astype(np.float32)
    
    # 3. Variable Roughness
    roughness_type = random.random()
    
    noise = np.zeros((h, w), dtype=np.float32)
    cv2.randu(noise, 0, 1.0)
    
    if roughness_type < 0.5:
        # Smooth edges (Liquid)
        noise = cv2.GaussianBlur(noise, (7, 7), 0)
        final_mask = np.where((mask - (noise * 0.2)) > 0.5, 1.0, 0.0)
    else:
        # Jagged edges (Crumb/Chip)
        final_mask = np.where((mask - (noise * 0.4)) > 0.5, 1.0, 0.0)
        
    return final_mask.astype(np.float32), stretch_angle

def add_micro_satellites(mask, cx, cy):
    """Adds small specs nearby."""
    h, w = mask.shape
    num_sats = random.randint(0, 6) 
    
    for _ in range(num_sats):
        # Scatter within slightly larger range (15-35px) due to larger main defect
        r = random.uniform(15, 35)
        theta = random.uniform(0, 2*np.pi)
        sx = int(cx + r * np.cos(theta))
        sy = int(cy + r * np.sin(theta))
        
        if 0 <= sx < w and 0 <= sy < h:
            # 1-2 pixel dots
            cv2.circle(mask, (sx, sy), random.randint(1, 2), 1.0, -1)
            
    return mask

def generate_small_cmp_defect(img):
    """Generates a medium-small, varied, opaque CMP defect."""
    h, w = img.shape[:2]
    canvas = img.astype(np.float32)
    bg_mean = np.mean(canvas)
    
    # 1. Random Location
    cx, cy = random.randint(40, w-40), random.randint(40, h-40)
    
    # 2. Generate Shape
    mask, motion_angle = generate_micro_cluster_shape(h, w, cx, cy)
    mask = add_micro_satellites(mask, cx, cy)
    
    # 3. Pattern Erosion
    erosion_zone = cv2.GaussianBlur(mask, (19, 19), 0) # Slightly larger erosion zone
    erosion_strength = 0.95 if random.random() < 0.05 else 0.7
    
    for c in range(3):
        canvas[:,:,c] = canvas[:,:,c] * (1 - erosion_zone * erosion_strength) + (bg_mean * erosion_zone * erosion_strength)

    # 4. Patch Gradient & Texture
    Y, X = np.ogrid[:h, :w]
    dist_along_axis = (X - cx) * np.cos(motion_angle) + (Y - cy) * np.sin(motion_angle)
    
    # Gradient
    mask_indices = mask > 0
    if np.any(mask_indices):
        g_min, g_max = dist_along_axis[mask_indices].min(), dist_along_axis[mask_indices].max()
        norm_grad = (dist_along_axis - g_min) / (g_max - g_min + 1e-5)
        intensity_grad = (norm_grad - 0.5) * random.uniform(-30, -10)
    else:
        intensity_grad = 0
        
    # Base Color
    patch_base = bg_mean + random.uniform(30, 60)
    grain = np.zeros((h, w), dtype=np.float32)
    cv2.randu(grain, -10, 10)
    
    patch_tex = patch_base + grain + intensity_grad

    # 5. 3D Shading
    height_map = cv2.GaussianBlur(mask, (7, 7), 0) 
    gx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)
    shading = (gx + gy) * 40
    patch_tex += shading

    # 6. Final Opaque Paste
    mask_3d = mask[:, :, np.newaxis]
    patch_3d = np.repeat(patch_tex[:, :, np.newaxis], 3, axis=2)
    
    # Shadow
    shadow = cv2.GaussianBlur(mask, (7, 7), 0) * 0.5
    canvas *= (1 - shadow[:, :, np.newaxis])
    
    canvas = (canvas * (1 - mask_3d)) + (patch_3d * mask_3d)

    # 7. Export
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    
    # Adjusted BBox padding
    coords = np.argwhere(mask > 0)
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        # Increased padding slightly for larger defects
        bbox = [0, cx/w, cy/h, (x_max - x_min + 8)/w, (y_max - y_min + 8)/h]
    else:
        bbox = [0, 0.5, 0.5, 0.01, 0.01]
    
    return canvas, bbox

def process_factory():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: {INPUT_FOLDER} folder missing.")
        return

    files = [os.path.join(INPUT_FOLDER, f) for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    if not files: 
        print("No background images found!")
        return

    print(f"Generating {SAMPLES_PER_CLASS} Medium-Small Varied CMP Defects...")

    for i in range(SAMPLES_PER_CLASS):
        roll = random.random()
        if roll < 0.70: num_defects = 1
        elif roll < 0.90: num_defects = 2
        else: num_defects = 3 + random.randint(0, 2)

        img_choice = random.choice(files)
        img = cv2.imread(img_choice)
        img = cv2.resize(img, TARGET_SIZE)
        
        bboxes = []
        for _ in range(num_defects):
            img, box = generate_small_cmp_defect(img)
            bboxes.append(box)

        name = f'CMP_MED_VAR_{i:04d}.png'
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, name), img)
        with open(os.path.join(LBL_OUTPUT_DIR, name.replace('.png', '.txt')), 'w') as f:
            for b in bboxes:
                f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    print(f"\nSuccess! Generated 2500 images.")

if __name__ == "__main__":
    process_factory()