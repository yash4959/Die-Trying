import cv2
import numpy as np
import random
import os

# --- CONFIGURATION ---
INPUT_FOLDER = 'Metal_LS_Base_Imgs'
OUTPUT_BASE = 'Staging_Dataset'
CATEGORY = 'CMP'
TARGET_SIZE = (224, 224)
SAMPLES_PER_CLASS = 1500  # Updated to 1500
LER_CLASS_ID = 0

# --- PATH SETUP ---
IMG_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'images')
LBL_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'labels')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

def create_3d_bump_layer(img, num_bumps):
    """Generates a high-quality topographic 3D swell without pixel stretching."""
    h, w = img.shape
    heightmap = np.zeros((h, w), dtype=np.float32)

    # 1. Orientation Detection (Vertical vs Horizontal lines)
    gx = np.abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)).mean()
    gy = np.abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)).mean()
    is_vertical = gx > gy

    # 2. Locate Via Walls for Anchoring
    profile = img.mean(axis=0) if is_vertical else img.mean(axis=1)
    grad = np.abs(np.gradient(profile))
    centers = np.where(grad > np.percentile(grad, 88))[0]
    if centers.size == 0: return img

    for _ in range(num_bumps):
        cx = random.choice(centers)
        cy = random.randint(60, (h if is_vertical else w) - 60)
        
        # Organic Spline Parameters (Smoother Curves)
        temp_layer = np.zeros((h, w), dtype=np.float32)
        if is_vertical:
            # Vertical line: Bump stretches vertically
            cv2.circle(temp_layer, (cx, cy), random.randint(6, 12), 1.0, -1)
            sigma_y, sigma_x = random.randint(20, 35), random.randint(8, 14)
        else:
            # Horizontal line: Bump stretches horizontally
            cv2.circle(temp_layer, (cy, cx), random.randint(6, 12), 1.0, -1)
            sigma_y, sigma_x = random.randint(8, 14), random.randint(20, 35)

        # Smooth the seed into a proper 3D dome
        heightmap += cv2.GaussianBlur(temp_layer, (0, 0), sigmaX=sigma_x, sigmaY=sigma_y)

    # --- 3. TOPOGRAPHIC RENDERING ---
    # Calculate surface slopes (normals) for 3D depth
    dzdx = cv2.Sobel(heightmap, cv2.CV_32F, 1, 0, ksize=5)
    dzdy = cv2.Sobel(heightmap, cv2.CV_32F, 0, 1, ksize=5)

    # Simulate SEM Light Source (Physical 3D highlight/shadow)
    # 
    light_map = (dzdx + dzdy) / 2.0
    # Apply a "Raised" protrusion factor
    shading = 1.0 + (light_map * 0.7) 
    
    # 4. Final Blend
    # This applies the 3D light to the original via texture without any holes
    result = img.astype(np.float32) * shading
    
    return np.clip(result, 0, 255).astype(np.uint8)

def process_factory():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' not found.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    if not files: return

    # Write class metadata
    with open(os.path.join(OUTPUT_BASE, CATEGORY, 'classes.txt'), 'w') as f:
        f.write("CMP_BUMP\n")

    print(f"Generating {SAMPLES_PER_CLASS} Perfect 3D CMP Bumps...")

    for count in range(SAMPLES_PER_CLASS):
        # 80% (1 bump), 15% (2 bumps), 5% (3 bumps)
        p = random.random()
        n = 1 if p < 0.80 else 2 if p < 0.95 else 3

        base_img = cv2.imread(os.path.join(INPUT_FOLDER, random.choice(files)), 0)
        canvas = cv2.resize(base_img, TARGET_SIZE)

        # Apply the Topographic Layer
        canvas = create_3d_bump_layer(canvas, n)

        # Save Image & YOLO formatted Label
        out_name = f'CMP_PERFECT_3D_{count:04d}.png'
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, out_name), canvas)
        
        with open(os.path.join(LBL_OUTPUT_DIR, out_name.replace('.png', '.txt')), 'w') as f:
            f.write("0 0.5 0.5 1.0 1.0\n")

        if count % 150 == 0:
            print(f"Status: {count}/{SAMPLES_PER_CLASS} generated...")

    print(f"\nSuccess! 1500 images created in {OUTPUT_BASE}/{CATEGORY}")

if __name__ == "__main__":
    process_factory()