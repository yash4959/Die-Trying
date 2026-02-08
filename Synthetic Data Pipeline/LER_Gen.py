import cv2
import numpy as np
import random
import os
import re

# --- CONFIGURATION ---
INPUT_FOLDER = 'Metal_LS_Base_Imgs'
OUTPUT_BASE = 'Staging_Dataset'
CATEGORY = 'LER_MED_HIGH_FREQ'
TARGET_SIZE = (224, 224)
SAMPLES_PER_CLASS = 2500
LER_CLASS_ID = 2

# --- PATH SETUP ---
IMG_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'images')
LBL_OUTPUT_DIR = os.path.join(OUTPUT_BASE, CATEGORY, 'labels')
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)
os.makedirs(LBL_OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---
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

# --- CORE LER ENGINE: MEDIUM-HIGH FREQUENCY ---
def apply_bumpy_ler_med_high(img):
    h, w = img.shape
    img_f = img.astype(np.float32)
    
    col_profile = img_f.mean(axis=0)
    grad = np.abs(np.gradient(col_profile))
    thresh = np.percentile(grad, 70) 
    line_x = np.where(grad > thresh)[0]
    
    if len(line_x) == 0: return img, [] 
    lines = np.split(line_x, np.where(np.diff(line_x) > 3)[0] + 1)
    line_centers = [int(np.mean(l)) for l in lines if len(l) > 2]

    if len(line_centers) < 1: return img, []

    # --- UPDATED PARAMETERS FOR SLIGHTLY HIGHER FREQUENCY ---
    amp_range = (2.8, 5.0)   # High visibility maintained
    wav_range = (25, 45)     # MEDIUM-HIGH: Faster bumps than low-freq, but still structured
    corr_width = 10          # Tightened correlation for sharper bumps
    
    y = np.arange(h)
    disp_field = np.zeros((h, w), dtype=np.float32)
    raw_boxes = [] 

    # --- VARIABLE DISTRIBUTION LOGIC (SAFE RANGE) ---
    dist_roll = random.random()
    num_detected = len(line_centers)

    if dist_roll < 0.33:
        target_count = random.randint(1, min(3, num_detected)) #
    elif dist_roll < 0.66:
        target_count = max(1, int(num_detected * 0.5)) #
    else:
        target_count = max(1, int(num_detected * random.uniform(0.9, 1.0))) #

    target_lines = random.sample(line_centers, target_count)

    for cx in target_lines:
        A = random.uniform(*amp_range)
        wav = random.uniform(*wav_range)
        ph = random.uniform(0, 2*np.pi)
        
        # Bumpy Wave Construction
        bumpy_wave = A * np.abs(np.sin(np.pi * y / wav + ph)) #
        jitter = (A/2.5) * np.sin(2 * np.pi * y / (wav/4)) # Proportional jitter
        total_disp = bumpy_wave + jitter
        
        for x in range(w):
            weight = np.exp(-abs(x - cx) / corr_width)
            disp_field[:, x] += total_disp * weight
            
        raw_boxes.append({'cx': cx, 'cy': h/2, 'w': (A * 6.0) + 25, 'h': h}) #

    disp_field = cv2.GaussianBlur(disp_field, (3, 3), 0)
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + disp_field
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    distorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return distorted_img, raw_boxes

def apply_final_sem_pass(img):
    img = cv2.bilateralFilter(img, 7, 30, 30) # Preserve bumpy edges while cleaning background
    avg_brightness = np.mean(img)
    adaptive_peak = 75 + (avg_brightness / 255.0) * 80 
    img_float = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_float * adaptive_peak) / adaptive_peak
    return (np.clip(noisy * 255, 0, 255)).astype(np.uint8)

def save_label(out_name, final_boxes, img_w, img_h):
    txt_name = out_name.rsplit('.', 1)[0] + ".txt"
    with open(os.path.join(LBL_OUTPUT_DIR, txt_name), 'w') as f:
        for box in final_boxes:
            f.write(f"{LER_CLASS_ID} {box['cx']/img_w:.6f} {box['cy']/img_h:.6f} {box['w']/img_w:.6f} {box['h']/img_h:.6f}\n") #

def process_factory():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files: 
        print(f"Error: No images found in {INPUT_FOLDER}")
        return

    print(f"Generating {SAMPLES_PER_CLASS} Med-High Frequency Bumpy LER samples...")
    count = 0
    while count < SAMPLES_PER_CLASS:
        fname = random.choice(files)
        base_img = cv2.imread(os.path.join(INPUT_FOLDER, fname), 0)
        if base_img is None: continue
        canvas = cv2.resize(base_img, TARGET_SIZE)
        h, w = canvas.shape

        angle = get_angle_from_filename(fname)
        vertical_canvas, _ = rotate_image(canvas, -angle)
        
        distorted_rot, raw_boxes = apply_bumpy_ler_med_high(vertical_canvas)
        
        if not raw_boxes: continue 

        final_canvas, M_inv = rotate_image(distorted_rot, angle)

        final_boxes = []
        for b in raw_boxes:
            corners = [(b['cx']-b['w']/2, 0), (b['cx']+b['w']/2, 0), (b['cx']+b['w']/2, h), (b['cx']-b['w']/2, h)]
            mapped = [rotate_point(p, M_inv) for p in corners]
            xs, ys = [p[0] for p in mapped], [p[1] for p in mapped]
            final_boxes.append({'cx': (min(xs)+max(xs))/2, 'cy': (min(ys)+max(ys))/2, 'w': max(xs)-min(xs), 'h': max(ys)-min(ys)})

        final_canvas = apply_final_sem_pass(final_canvas)
        
        out_name = f'LER_MED_HIGH_{count:05d}.png'
        cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, out_name), final_canvas)
        save_label(out_name, final_boxes, w, h)
        
        count += 1
        if count % 250 == 0: print(f"Progress: {count}/{SAMPLES_PER_CLASS}...")

    print(f"\nSuccess! Med-High Frequency Dataset created.")

if __name__ == "__main__":
    process_factory()