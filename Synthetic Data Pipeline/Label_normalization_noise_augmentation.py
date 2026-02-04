import os
import shutil
import random
import cv2
import numpy as np
from collections import defaultdict, Counter
from glob import glob

# ================= CONFIGURATION =================
SOURCE_ROOT = "Staging_Dataset"
DEST_ROOT = "NXP_Final_Submission"

# MENU OPTIONS
MENU = {
    '1': ['Bridge'],
    '2': ['Open'],
    '3': ['LER'],
    '4': ['CMP'],
    '5': ['Crack'],
    '6': ['Particle'], # Will map to 'Other'
    '7': ['Via'],
    '8': ['Clean'],
    '9': ['Patterning defects', 'Planarization defects'], 
    '0': 'ALL' 
}

# MAPPINGS
LEGACY_MAPPINGS = {
    'Patterning defects': {0: 0, 1: 1, 2: 3, 3: 2},
    'Planarization defects': {3: 3, 4: 4}
}

ID_TO_FOLDER = {
    0: 'Bridge', 1: 'Open', 2: 'LER', 3: 'CMP', 
    4: 'Crack', 5: 'Other', 6: 'Via'
}
# Reverse lookup for counting
FOLDER_TO_ID = {v: k for k, v in ID_TO_FOLDER.items()}

SPLIT_RATIOS = {'Train': 0.7, 'Validation': 0.2, 'Test': 0.1}

# ================= AUGMENTATION =================
def apply_sim_to_real_transform(img):
    if img is None: return None
    alpha = random.uniform(0.9, 1.1); beta = random.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    h, w = img.shape[:2]
    
    if random.random() < 0.8:
        sigma = random.uniform(3, 10)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        speckle = np.random.randn(h, w, 1) if len(img.shape)==3 else np.random.randn(h, w)
        img = np.clip(img + img * speckle * 0.08, 0, 255).astype(np.uint8)
    
    x = np.linspace(-1, 1, w); y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    mask = 1 - (np.sqrt(X**2 + Y**2) * random.uniform(0.15, 0.40))
    mask = np.clip(mask, 0.4, 1.0)
    if len(img.shape) == 3: mask = mask[..., None]
    img = (img.astype(np.float32) * mask).astype(np.uint8)
    
    if random.random() < 0.3: img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

# ================= HELPER: COUNT EXISTING =================
def scan_current_counts():
    """Checks the destination folder to see how unbalanced we currently are."""
    counts = Counter()
    print("📊 Scanning existing dataset state...")
    
    if not os.path.exists(DEST_ROOT):
        return counts

    for split in ['Train', 'Validation', 'Test']:
        split_path = os.path.join(DEST_ROOT, split)
        if not os.path.exists(split_path): continue
        
        for folder_name in os.listdir(split_path):
            # Map folder name back to ID (Bridge -> 0)
            if folder_name in FOLDER_TO_ID:
                fid = FOLDER_TO_ID[folder_name]
                # Count images
                path = os.path.join(split_path, folder_name)
                n = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg'))])
                counts[fid] += n
            elif folder_name == 'Clean':
                # We can track clean too if needed, but mostly we care about defect balance
                pass
                
    print(f"   Current Counts: {dict(counts)}")
    return counts

# ================= MAIN LOGIC =================
def process_dataset(target_folders=None):
    
    is_global_run = target_folders == 'ALL'
    
    # 1. INITIALIZE COUNTS
    if is_global_run:
        # If running ALL, we wipe and start fresh (counts = 0)
        if os.path.exists(DEST_ROOT): shutil.rmtree(DEST_ROOT)
        class_counts = Counter()
        folders_to_scan = os.listdir(SOURCE_ROOT)
        print("🌍 GLOBAL RESET: Wiping destination and rebalancing everything.")
    else:
        # If running Incremental, we READ existing counts
        class_counts = scan_current_counts()
        folders_to_scan = [f for f in target_folders if os.path.exists(os.path.join(SOURCE_ROOT, f))]
        print(f"🎯 INCREMENTAL MODE: Balancing new data against existing counts.")

    # 2. SCANNING & ROUTING
    final_routing = defaultdict(list)
    
    # We now put EVERYTHING into the balancing logic, even in incremental mode
    single_label_queue = []
    multi_label_queue = []
    clean_queue = []

    for folder in folders_to_scan:
        folder_path = os.path.join(SOURCE_ROOT, folder)
        if not os.path.isdir(folder_path): continue
        
        mapping_rule = LEGACY_MAPPINGS.get(folder, {})
        is_mixed = folder in LEGACY_MAPPINGS
        img_dir = os.path.join(folder_path, "images")
        lbl_dir = os.path.join(folder_path, "labels")
        
        if not os.path.exists(img_dir): continue
        print(f"   - Scanning Source: {folder}...")

        for file in os.listdir(img_dir):
            if not file.endswith(('.png', '.jpg')): continue
            
            img_path = os.path.join(img_dir, file)
            txt_file = file.replace('.png', '.txt').replace('.jpg', '.txt')
            lbl_path = os.path.join(lbl_dir, txt_file)

            # CLEAN
            if folder == 'Clean':
                clean_queue.append((img_path, None, None))
                continue

            # DEFECT
            if os.path.exists(lbl_path):
                found_ids = set()
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if not parts: continue
                        old_id = int(parts[0])
                        curr_id = mapping_rule.get(old_id, old_id) if is_mixed else old_id
                        if curr_id == 5: curr_id = 5 
                        if curr_id in ID_TO_FOLDER: found_ids.add(curr_id)
                
                found_ids = list(found_ids)
                if not found_ids: continue

                if len(found_ids) == 1:
                    single_label_queue.append((img_path, lbl_path, found_ids[0], mapping_rule))
                else:
                    multi_label_queue.append((img_path, lbl_path, found_ids, mapping_rule))

    # 3. APPLY SMART BALANCING
    print(f"⚖️  Routing {len(single_label_queue) + len(multi_label_queue)} images...")

    # A. Single Labels (Forced)
    for (img, lbl, fid, rule) in single_label_queue:
        fname = ID_TO_FOLDER[fid]
        final_routing[fname].append((img, lbl, rule))
        class_counts[fid] += 1 # Increment existing count

    # B. Multi Labels (Smart Choice based on TOTAL counts)
    for (img, lbl, fids, rule) in multi_label_queue:
        # Pick the class that currently has the FEWEST total images (Existing + New Single Labels)
        best_id = min(fids, key=lambda x: class_counts[x])
        
        fname = ID_TO_FOLDER[best_id]
        final_routing[fname].append((img, lbl, rule))
        class_counts[best_id] += 1

    # C. Clean
    for item in clean_queue:
        final_routing['Clean'].append(item)

    # 4. WRITING TO DISK
    print("🚀 Augmenting and Saving...")
    
    for class_name, items in final_routing.items():
        if not items: continue
        
        # Determine Split Counts
        n_total = len(items)
        n_train = int(n_total * SPLIT_RATIOS['Train'])
        n_val   = int(n_total * SPLIT_RATIOS['Validation'])
        
        splits = {
            'Train': items[:n_train],
            'Validation': items[n_train:n_train+n_val],
            'Test': items[n_train+n_val:]
        }
        
        print(f"   📂 {class_name}: Adding {n_total} new images.")
        
        for split_name, split_data in splits.items():
            dest_dir = os.path.join(DEST_ROOT, split_name, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for (src_img, src_lbl, rule) in split_data:
                fname = os.path.basename(src_img)
                
                # Check duplication (Don't overwrite if it exists from previous run)
                # Actually, in incremental mode, we usually WANT to add. 
                # To prevent filename collisions, we prepend a tag if needed, 
                # but standard filenames usually differ by source ID.
                
                # Augment
                img = cv2.imread(src_img)
                aug_img = apply_sim_to_real_transform(img)
                cv2.imwrite(os.path.join(dest_dir, fname), aug_img)
                
                # Label
                dest_txt = os.path.join(dest_dir, fname.replace('.png','.txt'))
                if class_name == 'Clean':
                    open(dest_txt, 'w').close()
                else:
                    if src_lbl and os.path.exists(src_lbl):
                        with open(src_lbl, 'r') as f_in, open(dest_txt, 'w') as f_out:
                            for line in f_in:
                                parts = line.strip().split()
                                old_id = int(parts[0])
                                final_id = rule.get(old_id, old_id) if rule else old_id
                                if final_id == 5: final_id = 5
                                parts[0] = str(final_id)
                                f_out.write(" ".join(parts) + "\n")

    print(f"✅ DONE. New totals: {dict(class_counts)}")

if __name__ == "__main__":
    print("\n--- INCREMENTAL LOAD BALANCER ---")
    print("1. Bridge")
    print("2. Open")
    print("3. LER")
    print("4. CMP")
    print("5. Crack")
    print("6. Particle (-> Other)")
    print("7. Via")
    print("8. Clean")
    print("9. Mixed Folders (Patterning/Planarization)")
    print("0. ALL (Reset & Full Balance)")
    
    choice = input("\nEnter choice (0-9): ").strip()
    
    if choice in MENU:
        process_dataset(MENU[choice])
    else:
        print("❌ Invalid choice")