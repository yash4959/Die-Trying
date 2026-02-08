import os
import shutil
import random
import cv2
import numpy as np
from collections import defaultdict, Counter

# ================= CONFIGURATION =================
SOURCE_ROOT = "Staging_Dataset"
DEST_ROOT = "NXP_Final_Submission"

# TARGET CLASS MAP
ID_TO_FOLDER = {
    0: 'Bridge', 1: 'Open', 2: 'LER', 3: 'CMP', 
    4: 'Crack', 5: 'Other', 6: 'Via'
}

# FOLDER MAPPINGS
FOLDER_MAPPINGS = {
    'Bridge':   {0: 0},
    'Open':     {1: 1},
    'LER':      {2: 2},
    'Particle': {6: 5},
    'CMP':      {0: 3, 3: 3},
    'Via':      {0: 6, 6: 6},
    'Crack':    {0: 4, 3: 4, 4: 4}, 
    'Planarization defects': {3: 3, 4: 4, 6: 5},
    'Patterning Defects':    {0: 0, 1: 1, 2: 2, 3: 3, 6: 5},
    'Via Array Defects':     {0: 6, 1: 5, 2: 3},
    'Mixed_Metal_LS':        {0: 0, 1: 1, 2: 2, 3: 3, 6: 5},
    'Mixed_Defects':         {3: 3, 4: 4, 6: 5} 
}

SPLIT_RATIOS = {'Train': 0.7, 'Validation': 0.2, 'Test': 0.1}

# ================= AUGMENTATION (THE EQUALIZER) =================

def add_heavy_salt_and_pepper(image, prob=0.05):
    """
    Adds distinct S&P noise to force the model to look past the static.
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        black = np.array([0, 0, 0], dtype='uint8')
        white = np.array([255, 255, 255], dtype='uint8')

    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def apply_sim_to_real_transform(img, class_label):
    if img is None: return None
    if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape

    # 1. Base Realism (Lighting/Contrast) - Apply to ALL
    alpha = random.uniform(0.9, 1.1)
    beta = random.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # 2. Gaussian Noise (Sensor Grain) - Apply to ALL
    if random.random() < 0.8:
        sigma = random.uniform(3, 8)
        noise = np.random.normal(0, sigma, (h, w)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    # 3. Vignette (Microscope Edge) - Apply to ALL
    x = np.linspace(-1, 1, w); y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    mask = 1 - (np.sqrt(X**2 + Y**2) * random.uniform(0.15, 0.40))
    img = (img.astype(np.float32) * np.clip(mask, 0.4, 1.0)).astype(np.uint8)
    
    # --- THE EQUALIZER LOGIC ---
    # IF the class is NOT Bridge/Open, we force S&P noise onto it.
    # This prevents the model from thinking "Noise = Defect".
    if class_label not in ['Bridge', 'Open']:
        # We apply S&P noise to Clean, CMP, Via, etc.
        # Probability 0.04 matches the intensity of your generated defects
        img = add_heavy_salt_and_pepper(img, prob=0.04)
    
    # 4. Blur (Focus drift) - Apply to ALL
    if random.random() < 0.3: img = cv2.GaussianBlur(img, (3, 3), 0)
    
    return img

# ================= PROCESSING LOGIC =================
def process_dataset(target_list):
    if target_list == 'ALL':
        folders_to_scan = [f for f in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, f))]
        if os.path.exists(DEST_ROOT): shutil.rmtree(DEST_ROOT)
    else:
        folders_to_scan = [f for f in target_list if os.path.exists(os.path.join(SOURCE_ROOT, f))]

    print(f"📂 Processing: {folders_to_scan}")
    
    # 1. SCAN & QUEUE
    single_queue = []
    mixed_queue  = []
    clean_queue  = []
    
    for folder in folders_to_scan:
        img_dir = os.path.join(SOURCE_ROOT, folder, 'images')
        lbl_dir = os.path.join(SOURCE_ROOT, folder, 'labels')
        if not os.path.exists(img_dir): continue
        
        mapping = FOLDER_MAPPINGS.get(folder, {})
        files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        
        for fname in files:
            img_p = os.path.join(img_dir, fname)
            txt_p = os.path.join(lbl_dir, fname.replace('.png','.txt').replace('.jpg','.txt'))
            
            found_classes = set()
            new_lines = []
            
            if folder == 'Clean':
                found_classes.add('Clean')
            elif os.path.exists(txt_p):
                with open(txt_p, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if not parts: continue
                    raw_id = int(parts[0])
                    final_id = mapping.get(raw_id, raw_id) 
                    
                    parts[0] = str(final_id)
                    new_lines.append(" ".join(parts))
                    if final_id in ID_TO_FOLDER:
                        found_classes.add(ID_TO_FOLDER[final_id])
            
            if not found_classes: continue
            
            task = {'img': img_p, 'lines': new_lines, 'classes': list(found_classes)}
            
            if 'Clean' in found_classes: clean_queue.append(task)
            elif len(found_classes) == 1: single_queue.append(task)
            else: mixed_queue.append(task)

    # 2. SAVE LOGIC
    stats = defaultdict(int)
    
    def save_file(task, primary_class):
        r = random.random()
        if r < SPLIT_RATIOS['Train']: split = 'Train'
        elif r < SPLIT_RATIOS['Train'] + SPLIT_RATIOS['Validation']: split = 'Validation'
        else: split = 'Test'
        
        d_img = os.path.join(DEST_ROOT, split, primary_class, 'images')
        d_lbl = os.path.join(DEST_ROOT, split, primary_class, 'labels')
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)
        
        fname = os.path.basename(task['img'])
        img = cv2.imread(task['img'])
        
        # *** PASS THE CLASS LABEL HERE ***
        aug = apply_sim_to_real_transform(img, primary_class)
        
        cv2.imwrite(os.path.join(d_img, fname), aug)
        
        with open(os.path.join(d_lbl, fname.replace('.png','.txt')), 'w') as f:
            f.write("\n".join(task['lines']))
            
        stats[primary_class] += 1

    print(f"📊 Queues: {len(single_queue)} Single | {len(mixed_queue)} Mixed | {len(clean_queue)} Clean")
    
    random.shuffle(single_queue)
    for task in single_queue:
        save_file(task, task['classes'][0])
        
    print(f"   Baseline Counts: {dict(stats)}")
    
    random.shuffle(mixed_queue)
    for task in mixed_queue:
        candidates = task['classes'] 
        candidates.sort(key=lambda c: stats[c]) 
        save_file(task, candidates[0])

    for task in clean_queue[:1500]: 
        save_file(task, 'Clean')

    print("\n✅ DONE. Final Dataset Balance:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    print("--- DATASET BUILDER (NOISE EQUALIZER) ---")
    c = input("Type '0' to run ALL: ").strip()
    if c == '0': process_dataset('ALL')