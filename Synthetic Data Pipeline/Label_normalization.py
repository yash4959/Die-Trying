import os
import shutil
import random
import yaml

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_ROOT = "Staging_Dataset"
DEST_ROOT = "Final_Normalized_Dataset"

# SPLIT RATIOS (Must sum to 1.0)
TRAIN_RATIO = 0.7 
VAL_RATIO   = 0.2 
# Test is remainder

CLASS_NAMES = [
    'Bridge',       # 0
    'Open',         # 1
    'LER',          # 2
    'CMP',          # 3
    'Crack',        # 4
    'Particle',     # 5
    'Via'           # 6
]

FOLDER_RULES = {
    # SPECIAL HANDLING FOR CLEAN IMAGES
    'Clean': {'action': 'clean'}, 

    # MAPPING RULES
    'Patterning defects': {
        'action': 'map',
        'mapping': {0: 0, 1: 1, 2: 3, 3: 2} 
    },
    'Planarization defects': {
        'action': 'map',
        'mapping': {3: 3, 4: 4}
    },
    # FORCE RULES
    'Bridge':   {'action': 'force', 'id': 0},
    'Open':     {'action': 'force', 'id': 1},
    'LER':      {'action': 'force', 'id': 2},
    'CMP':      {'action': 'force', 'id': 3},
    'Crack':    {'action': 'force', 'id': 4},
    'Particle': {'action': 'force', 'id': 5},
    'Via':      {'action': 'force', 'id': 6}
}

def setup_dirs():
    if os.path.exists(DEST_ROOT):
        shutil.rmtree(DEST_ROOT)
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{DEST_ROOT}/{split}/images", exist_ok=True)
        os.makedirs(f"{DEST_ROOT}/{split}/labels", exist_ok=True)

def normalize_and_merge():
    print(f"--- 🚀 STARTING NORMALIZATION (Including Clean Images) ---")
    setup_dirs()
    
    total_images = 0
    if not os.path.exists(SOURCE_ROOT): return

    available_folders = [d for d in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, d))]
    
    for folder in available_folders:
        rule = None
        for r_name in FOLDER_RULES:
            if r_name.lower().replace('_', ' ') == folder.lower().replace('_', ' '):
                rule = FOLDER_RULES[r_name]
                break
        
        if not rule: continue
        print(f"📂 Processing {folder}...")
        
        img_src = os.path.join(SOURCE_ROOT, folder, 'images')
        lbl_src = os.path.join(SOURCE_ROOT, folder, 'labels') # Might not exist for Clean
        
        if not os.path.exists(img_src): continue
        files = os.listdir(img_src)
        
        for file in files:
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            src_img_path = os.path.join(img_src, file)
            
            # --- 3-WAY SPLIT LOGIC ---
            rand = random.random()
            if rand < TRAIN_RATIO: split = 'train'
            elif rand < (TRAIN_RATIO + VAL_RATIO): split = 'val'
            else: split = 'test'
            
            clean_folder_name = folder.replace(' ', '_')
            dest_filename = f"{clean_folder_name}_{file}"
            dest_txt_name = dest_filename.rsplit('.', 1)[0] + ".txt"

            # === HANDLING CLEAN IMAGES ===
            if rule['action'] == 'clean':
                # Copy Image
                shutil.copy(src_img_path, f"{DEST_ROOT}/{split}/images/{dest_filename}")
                # Create EMPTY Text File (Crucial for YOLO to know it's a negative sample)
                open(f"{DEST_ROOT}/{split}/labels/{dest_txt_name}", 'w').close()
                total_images += 1
                continue

            # === HANDLING DEFECT IMAGES ===
            txt_name = os.path.splitext(file)[0] + ".txt"
            src_txt_path = os.path.join(lbl_src, txt_name)
            
            if not os.path.exists(src_txt_path): continue
            
            new_lines = []
            valid_file = False
            
            with open(src_txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    old_id = int(parts[0])
                    
                    final_id = -1
                    if rule['action'] == 'force':
                        final_id = rule['id']
                    elif rule['action'] == 'map':
                        final_id = rule['mapping'].get(old_id, -1)
                    
                    if 0 <= final_id < len(CLASS_NAMES):
                        new_lines.append(f"{final_id} {' '.join(parts[1:])}\n")
                        valid_file = True

            if valid_file:
                shutil.copy(src_img_path, f"{DEST_ROOT}/{split}/images/{dest_filename}")
                with open(f"{DEST_ROOT}/{split}/labels/{dest_txt_name}", 'w') as f:
                    f.writelines(new_lines)
                total_images += 1

    # UPDATE YAML
    yaml_data = {
        'path': f"../dataset/{DEST_ROOT}",
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    
    with open(f"{DEST_ROOT}/data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"✅ DONE! {total_images} images processed.")
    print(f"📊 Split: 70/20/10 | Clean Images Included")

if __name__ == "__main__":
    normalize_and_merge()