import cv2
import numpy as np
import os
import random

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER  = 'Staging_Dataset/Clean/images'
OUTPUT_FOLDER = 'Staging_Dataset/Via/images'
LABEL_FOLDER  = 'Staging_Dataset/Via/labels'

TARGET_COUNT = 2500
CLASS_ID_MISSING_VIA = 0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(LABEL_FOLDER, exist_ok=True)

def setup_blob_detector():
    # Robust detector settings
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.65
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    return cv2.SimpleBlobDetector_create(params)

def generate_missing_vias():
    print(f"--- Generating {TARGET_COUNT} MISSING VIAS (V166: Smooth Multi-Healer) ---")
    
    detector = setup_blob_detector()
    
    all_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    via_files = [f for f in all_files if 'bg_via' in f]
    
    random.shuffle(via_files)
    selection = (via_files * (TARGET_COUNT // len(via_files) + 1))[:TARGET_COUNT]

    count = 0
    for filename in selection:
        img = cv2.imread(os.path.join(INPUT_FOLDER, filename))
        if img is None: continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Detect Vias
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        keypoints = detector.detect(enhanced_gray)

        if not keypoints: continue
        
        # --- LOGIC UPDATE: MULTI-TARGET SELECTION ---
        # If the image is dense (lots of vias), remove 2-4 of them.
        # If it's sparse, remove 1-2.
        total_vias = len(keypoints)
        if total_vias > 12:
            num_targets = random.randint(2, 4)
        elif total_vias > 4:
            num_targets = random.randint(1, 2)
        else:
            num_targets = 1
            
        # Ensure we don't try to remove more than exist
        num_targets = min(num_targets, total_vias)
        targets = random.sample(keypoints, num_targets)

        output = img.copy()
        labels = []
        
        # We create one master mask for all holes to heal them simultaneously
        overall_mask = np.zeros((h, w), dtype=np.uint8)
        valid_targets = []

        for kp in targets:
            cx, cy = int(kp.pt[0]), int(kp.pt[1])
            r = int(kp.size / 2)
            
            if r < 2 or cx-r < 0 or cy-r < 0 or cx+r >= w or cy+r >= h: continue
            
            # Expansion (1.6x) to kill the White Halo
            defect_r = int(r * 1.6)
            
            # Add to mask
            cv2.circle(overall_mask, (cx, cy), defect_r, 255, -1)
            valid_targets.append((cx, cy, r))

        if not valid_targets: continue

        # --- 2. SMOOTH HEALING (INPAINTING) ---
        # Telea works best for "smooth" gradients.
        output = cv2.inpaint(output, overall_mask, 3, cv2.INPAINT_TELEA)

        # --- 3. REALISTIC GRAIN (SOFTENED) ---
        # Instead of sharp static, we want soft SEM grain.
        
        # A. Generate noise
        noise = np.zeros(img.shape, dtype=np.int16)
        # Reduced intensity slightly (15 instead of 20) for smoother look
        cv2.randn(noise, (0,0,0), (15, 15, 15))
        
        # B. SOFTEN the noise
        # This is the "Smooth" trick. We blur the noise itself slightly.
        # It makes the grain look "analog" instead of "digital."
        noise_smooth = cv2.GaussianBlur(noise, (3, 3), 0)
        
        # C. Add noise to image
        output_int = output.astype(np.int16)
        output_noisy = cv2.add(output_int, noise_smooth)
        output_noisy = np.clip(output_noisy, 0, 255).astype(np.uint8)
        
        # D. Apply ONLY to the healed patches
        mask_bool = overall_mask > 0
        output[mask_bool] = output_noisy[mask_bool]

        # --- 4. BLEND EDGES ---
        # One final pass to ensure the "seam" between healed area and real image is invisible.
        # We apply a tiny blur just on the boundary of the mask.
        mask_edge = cv2.Canny(overall_mask, 100, 200)
        mask_edge = cv2.dilate(mask_edge, np.ones((3,3), np.uint8))
        # Where the edge is, we average the output to soften the transition
        output_blurred = cv2.GaussianBlur(output, (3,3), 0)
        output[mask_edge > 0] = output_blurred[mask_edge > 0]

        # --- 5. SAVE LABELS ---
        for (cx, cy, r) in valid_targets:
            norm_x = cx / w
            norm_y = cy / h
            norm_w = (r * 2) / w
            norm_h = (r * 2) / h
            labels.append(f"{CLASS_ID_MISSING_VIA} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        # Save to disk
        out_filename = f"miss_via_{count:06d}.png"
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, out_filename), output)
        with open(os.path.join(LABEL_FOLDER, f"miss_via_{count:06d}.txt"), "w") as f:
            f.writelines(labels)
            
        count += 1
        if count % 200 == 0: print(f"  Processed {count}/{TARGET_COUNT}...")

    print(f"\nDONE. Generated {count} smooth, realistic defects.")

if __name__ == "__main__":
    generate_missing_vias()