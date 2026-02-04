import cv2
import numpy as np
import random
import os
import re

# --- CONFIGURATION ---
INPUT_FOLDER = 'Metal_LS_Base_Imgs'
OUTPUT_BASE = 'Staging_Dataset'
CATEGORY = 'LER'
TARGET_SIZE = (224, 224)
SAMPLES_PER_CLASS = 2500
LER_CLASS_ID = 2

# --- PATH SETUP ---
IMG_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'images')
LBL_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'labels')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

# --- HELPER: ROTATION ---
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated, M

def rotate_point(point, M):
    x, y = point
    new_x = (M[0, 0] * x) + (M[0, 1] * y) + M[0, 2]
    new_y = (M[1, 0] * x) + (M[1, 1] * y) + M[1, 2]
    return (new_x, new_y)

def get_angle_from_filename(filename):
    match = re.search(r'deg(\d+)', filename)
    if match: return int(match.group(1))
    return 0

# --- CORE LER LOGIC (Selective Roughness) ---
def apply_vertical_ler(img, amp_range=(0.5, 1.5), wavelength_range=(5, 15)):
    h, w = img.shape
    img_f = img.astype(np.float32)
    
    # 1. Detect Vertical Lines
    col_profile = img_f.mean(axis=0)
    grad = np.abs(np.gradient(col_profile))
    thresh = np.percentile(grad, 75)
    line_x = np.where(grad > thresh)[0]
    
    lines = np.split(line_x, np.where(np.diff(line_x) > 3)[0] + 1)
    line_centers = [int(np.mean(l)) for l in lines if len(l) > 3]

    if len(line_centers) < 1: return img, []

    # 2. Select Targets
    num_defects = random.randint(1, min(3, len(line_centers)))
    target_lines = random.sample(line_centers, num_defects)
    
    y = np.arange(h)
    disp_field = np.zeros((h, w), dtype=np.float32)
    raw_boxes = [] 
    correlation_width = 8 

    # We need a mask to know EXACTLY which columns are "The Defect"
    defect_zone_mask = np.zeros((h, w), dtype=np.uint8)

    for cx in target_lines:
        A = random.uniform(*amp_range)
        wav = random.uniform(*wavelength_range)
        ph = random.uniform(0, 2*np.pi)
        
        # Jitter Wave
        total_disp = A * np.sin(2 * np.pi * y / wav + ph) + (A/2) * np.sin(2 * np.pi * y / (wav/2.5))
        
        # Update Displacement Field
        for x in range(w):
            weight = np.exp(-abs(x - cx) / correlation_width)
            disp_field[:, x] += total_disp * weight
            
        # Update Defect Zone Mask (Mark the area around the line)
        # We mark 10px on either side of the center line as "Active Zone"
        x_start = max(0, cx - 10)
        x_end = min(w, cx + 10)
        defect_zone_mask[:, x_start:x_end] = 255

        box_w = (A * 2.0) + 24 
        raw_boxes.append({'cx': cx, 'cy': h/2, 'w': box_w, 'h': h})

    # 3. Apply Geometric Remap
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + disp_field
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # 4. INJECT EDGE NOISE (Masked!)
    # We pass the defect_zone_mask so noise ONLY hits those specific lines
    distorted_img = inject_selective_edge_noise(distorted_img, defect_zone_mask)

    return distorted_img, raw_boxes

def inject_selective_edge_noise(img, zone_mask):
    """Adds heavy grain ONLY to the edges that reside inside the zone_mask."""
    
    # 1. Find ALL edges in the image
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    _, all_edges = cv2.threshold(abs_grad_x, 30, 255, cv2.THRESH_BINARY)
    
    # 2. INTERSECT: Global Edges AND Defect Zone
    # This kills any edge detection on the "clean" background lines
    target_edges = cv2.bitwise_and(all_edges, zone_mask)
    
    # 3. Dilate (Thicken) the target edges for visibility
    kernel = np.ones((3,3), np.uint8)
    dilated_target = cv2.dilate(target_edges, kernel, iterations=1)
    
    # 4. Generate & Apply Noise
    noise_sigma = 50 # Cranked up for visibility
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
    
    img_float = img.astype(np.float32)
    img_float[dilated_target > 0] += noise[dilated_target > 0]
    
    return np.clip(img_float, 0, 255).astype(np.uint8)

# --- POST-PROCESSING ---
def apply_3d_shadowing(img, intensity=0.15):
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    shadow_map = 1.0 + (grad_x / 255.0) * intensity
    out = img.astype(np.float32) * shadow_map
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_adaptive_noise(img):
    avg_brightness = np.mean(img)
    brightness_factor = avg_brightness / 255.0
    adaptive_peak = 12 + (brightness_factor * 43)
    img_float = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_float * adaptive_peak) / adaptive_peak
    out = (np.clip(noisy * 255, 0, 255)).astype(np.uint8)
    return out

def save_label(out_name, final_boxes, img_w, img_h):
    txt_name = out_name.rsplit('.', 1)[0] + ".txt"
    with open(os.path.join(LBL_OUTPUT_DIR, txt_name), 'w') as f:
        for box in final_boxes:
            nx = box['cx'] / img_w
            ny = box['cy'] / img_h
            nw = box['w'] / img_w
            nh = box['h'] / img_h
            
            nx = min(max(nx, 0), 1)
            ny = min(max(ny, 0), 1)
            nw = min(max(nw, 0), 1)
            nh = min(max(nh, 0), 1)

            f.write(f"{LER_CLASS_ID} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

def process_factory():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: 
        print(f"Error: No images found in {INPUT_FOLDER}")
        return

    print(f"Generating {SAMPLES_PER_CLASS} Selective LER images...")

    count = 0
    while count < SAMPLES_PER_CLASS:
        fname = random.choice(files)
        
        base_img = cv2.imread(os.path.join(INPUT_FOLDER, fname), 0)
        canvas = cv2.resize(base_img, TARGET_SIZE)
        h, w = canvas.shape

        angle = get_angle_from_filename(fname)
        rot_angle = -angle 
        vertical_canvas, M_fwd = rotate_image(canvas, rot_angle)

        distorted_rot, raw_boxes = apply_vertical_ler(vertical_canvas)
        
        if not raw_boxes: continue 

        final_canvas, M_inv = rotate_image(distorted_rot, -rot_angle)

        final_boxes = []
        for b in raw_boxes:
            x1, x2 = b['cx'] - b['w']/2, b['cx'] + b['w']/2
            y1, y2 = 0, h
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            mapped_corners = [rotate_point(p, M_inv) for p in corners]
            xs = [p[0] for p in mapped_corners]
            ys = [p[1] for p in mapped_corners]
            
            final_boxes.append({
                'cx': (min(xs) + max(xs)) / 2,
                'cy': (min(ys) + max(ys)) / 2,
                'w': max(xs) - min(xs),
                'h': max(ys) - min(ys)
            })

        final_canvas = apply_3d_shadowing(final_canvas)
        final_canvas = apply_adaptive_noise(final_canvas)
        if np.mean(final_canvas) < 85:
            final_canvas = cv2.GaussianBlur(final_canvas, (3, 3), 0.5)

        out_name = f'LER_SEL_{count:05d}.png'
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, out_name), final_canvas)
        save_label(out_name, final_boxes, w, h)

        count += 1
        if count % 250 == 0:
            print(f"Progress: {count}/{SAMPLES_PER_CLASS}...")

    print(f"\nSuccess! Dataset created in {OUTPUT_BASE}")

if __name__ == "__main__":
    process_factory()