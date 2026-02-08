import cv2
import numpy as np
import random
import os
import re
from scipy.signal import find_peaks

# ================= CONFIGURATION =================
INPUT_FOLDER = 'Metal_LS_Base_Imgs'
OUTPUT_ROOT = 'Staging_Dataset'

TARGET_COUNT = 1000
TARGET_SIZE = (224, 224)

# --- CLASS MAP ---
CLASS_IDS = {
    'BRIDGE': 0,
    'OPEN': 1,
    'LER': 2,
    'CMP': 3,
    'PARTICLE': 6
}

# Ensure Dirs
os.makedirs(f'{OUTPUT_ROOT}/Patterning Defects/images', exist_ok=True)
os.makedirs(f'{OUTPUT_ROOT}/Patterning Defects/labels', exist_ok=True)

# ================= 1. SHARED UTILS =================

def apply_global_grit(image):
    """Final SEM noise pass."""
    if len(image.shape) == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image.copy()
    gray = gray.astype(np.float32)
    h, w = gray.shape
    
    gaussian_noise = np.random.normal(0, 8, (h, w)).astype(np.float32)
    noisy_gray = gray + gaussian_noise
    
    salt_mask = np.random.random((h, w)) < 0.001
    noisy_gray[salt_mask] = 255
    pepper_mask = np.random.random((h, w)) < 0.001
    noisy_gray[pepper_mask] = 0
    
    final_gray = np.clip(noisy_gray, 0, 255).astype(np.uint8)
    return cv2.merge([final_gray, final_gray, final_gray])

def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT), M

def get_angle_from_filename(filename):
    match = re.search(r'deg(\d+)', filename)
    if match: return int(match.group(1))
    return 0

def get_texture_patch(w, h, mean_val=128):
    tex = np.zeros((h, w), dtype=np.float32)
    tex[:] = mean_val
    noise = np.random.normal(0, 15, (h, w))
    return np.clip(tex + noise, 0, 255).astype(np.uint8)

# ================= 2. LER ENGINE (ROTATION AWARE) =================

def apply_ler_distortion(img):
    """Warps valid vertical lines."""
    h, w = img.shape[:2]
    if len(img.shape) == 3: gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: gray_img = img
    
    col_profile = gray_img.mean(axis=0)
    grad = np.abs(np.gradient(col_profile))
    thresh = np.percentile(grad, 70) 
    line_x = np.where(grad > thresh)[0]
    if len(line_x) == 0: return img, []

    lines = np.split(line_x, np.where(np.diff(line_x) > 3)[0] + 1)
    line_centers = [int(np.mean(l)) for l in lines if len(l) > 2]
    if not line_centers: return img, []

    amp_range = (2.0, 4.0)
    wav_range = (20, 40)
    disp_field = np.zeros((h, w), dtype=np.float32)
    bboxes = []

    target_count = max(1, int(len(line_centers) * 0.5))
    targets = random.sample(line_centers, target_count)

    y = np.arange(h)
    for cx in targets:
        A = random.uniform(*amp_range)
        wav = random.uniform(*wav_range)
        ph = random.uniform(0, 2*np.pi)
        
        total_disp = A * np.abs(np.sin(np.pi * y / wav + ph))
        
        for x in range(w):
            weight = np.exp(-abs(x - cx) / 8.0)
            disp_field[:, x] += total_disp * weight
            
        # Box: center_x, center_y, width, height
        bboxes.append((CLASS_IDS['LER'], cx, h/2, (A*6.0)+20, h))

    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + disp_field
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))
    distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return distorted, bboxes

def calculate_rotated_aabb(cx, cy, w, h, angle, img_w, img_h):
    """Calculates the new Axis-Aligned Bounding Box (AABB) after rotation."""
    # Rotation Matrix
    M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
    
    # Define 4 corners of the original box
    x1, x2 = cx - w/2, cx + w/2
    y1, y2 = cy - h/2, cy + h/2
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ])
    
    # Rotate points
    ones = np.ones(shape=(len(corners), 1))
    points_ones = np.hstack([corners, ones])
    transformed_points = M.dot(points_ones.T).T
    
    # Find new min/max
    min_x = np.min(transformed_points[:, 0])
    max_x = np.max(transformed_points[:, 0])
    min_y = np.min(transformed_points[:, 1])
    max_y = np.max(transformed_points[:, 1])
    
    # Clamp to image bounds
    min_x = max(0, min_x); max_x = min(img_w, max_x)
    min_y = max(0, min_y); max_y = min(img_h, max_y)
    
    new_w = max_x - min_x
    new_h = max_y - min_y
    new_cx = min_x + new_w/2
    new_cy = min_y + new_h/2
    
    return new_cx, new_cy, new_w, new_h

# ================= 3. BRIDGE ENGINE =================

def warp_kissing_lines(img, p1, p2):
    h, w = img.shape[:2]
    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    gap_vec = np.array(p2) - np.array(p1)
    gap_dist = np.linalg.norm(gap_vec)
    if gap_dist < 1: return img, None

    radius = int(gap_dist * 1.5)
    x_min, x_max = max(0, cx-radius), min(w, cx+radius)
    y_min, y_max = max(0, cy-radius), min(h, cy+radius)
    if x_max <= x_min or y_max <= y_min: return img, None

    roi = img[y_min:y_max, x_min:x_max]
    rh, rw = roi.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(rw), np.arange(rh))
    
    rel_x = grid_x - (cx - x_min)
    rel_y = grid_y - (cy - y_min)
    dist_sq = rel_x**2 + rel_y**2
    force = np.exp(-dist_sq / (radius**2 * 0.5)) * (radius * 0.4)
    
    angle = np.arctan2(rel_y, rel_x)
    shift_x = np.cos(angle) * force
    shift_y = np.sin(angle) * force
    
    map_x = (grid_x + shift_x).astype(np.float32)
    map_y = (grid_y + shift_y).astype(np.float32)
    
    warped = cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img[y_min:y_max, x_min:x_max] = warped
    
    bbox = (CLASS_IDS['BRIDGE'], cx, cy, radius, radius)
    return img, bbox

def draw_drawn_bridge(img, p1, p2, thick):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.line(mask, p1, p2, 255, thick)
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv2.circle(mask, mid, thick+2, 255, -1)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines_mean = np.mean(gray[gray > 100]) if np.any(gray>100) else 150
    tex = get_texture_patch(img.shape[1], img.shape[0], lines_mean)
    
    mask_blur = cv2.GaussianBlur(mask, (3,3), 0)
    alpha = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    tex_bgr = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    
    final = (img.astype(float) * (1-alpha) + tex_bgr.astype(float) * alpha).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return final, (CLASS_IDS['BRIDGE'], bx+bw/2, by+bh/2, bw, bh)

# ================= 4. OPEN ENGINE =================

def draw_open_defect(img, center, lw, gw, u_line, u_gap):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    break_size = int(lw * random.uniform(0.4, 0.8))
    p1 = center - (u_gap * lw/2) - (u_line * break_size/2)
    p2 = center + (u_gap * lw/2) - (u_line * break_size/2)
    p3 = center + (u_gap * lw/2) + (u_line * break_size/2)
    p4 = center - (u_gap * lw/2) + (u_line * break_size/2)
    
    poly = np.array([p1, p2, p3, p4], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gap_mean = np.mean(gray[gray < 80]) if np.any(gray<80) else 30
    tex = get_texture_patch(img.shape[1], img.shape[0], gap_mean)
    
    mask_blur = cv2.GaussianBlur(mask, (3,3), 0)
    alpha = cv2.cvtColor(mask_blur, cv2.COLOR_GRAY2BGR).astype(float) / 255.0
    tex_bgr = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    
    final = (img.astype(float) * (1-alpha) + tex_bgr.astype(float) * alpha).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return final, (CLASS_IDS['OPEN'], bx+bw/2, by+bh/2, bw, bh)

# ================= 5. CMP & PARTICLE =================

def apply_cmp_scratch(img):
    h, w = img.shape[:2]
    canvas = img.astype(np.float32)
    cx, cy = random.randint(30, w-30), random.randint(30, h-30)
    
    mask = np.zeros((h, w), dtype=np.float32)
    axis_len = random.randint(5, 15)
    cv2.ellipse(mask, (cx, cy), (axis_len, random.randint(2, 5)), random.uniform(0, 180), 0, 360, 1.0, -1)
    
    for _ in range(3):
        sx = cx + random.randint(-15, 15); sy = cy + random.randint(-15, 15)
        cv2.circle(mask, (sx, sy), 1, 1.0, -1)
        
    mask = cv2.GaussianBlur(mask, (3,3), 0)
    noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
    canvas *= (1 - mask[:,:,None] * 0.4)
    canvas += (mask[:,:,None] * noise[:,:,None] * 0.5)
    final = np.clip(canvas, 0, 255).astype(np.uint8)
    
    coords = np.argwhere(mask > 0.1)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
        return final, (CLASS_IDS['CMP'], (x0+x1)/2, (y0+y1)/2, x1-x0, y1-y0)
    return img, None

def apply_particle(img):
    h, w = img.shape[:2]
    cx, cy = random.randint(20, w-20), random.randint(20, h-20)
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = random.randint(4, 10)
    
    pts = []
    for i in range(10):
        ang = (i/10)*2*np.pi; r = radius * random.uniform(0.8, 1.2)
        pts.append((int(cx + r*np.cos(ang)), int(cy + r*np.sin(ang))))
    cv2.fillPoly(mask, [np.array(pts)], 255)
    
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0: dist = dist/dist.max()
    core = 200 + (dist * 55) + np.random.normal(0, 10, (h,w))
    
    mask_blur = cv2.GaussianBlur(mask, (3,3), 0)
    alpha = mask_blur.astype(float)/255.0; alpha = alpha[:,:,None]
    particle_rgb = cv2.merge([core, core, core])
    
    final = (img.astype(float)*(1-alpha) + particle_rgb*alpha).astype(np.uint8)
    bx, by, bw, bh = cv2.boundingRect(mask)
    return final, (CLASS_IDS['PARTICLE'], bx+bw/2, by+bh/2, bw, bh)

# ================= 6. LINE COORDINATES =================

def get_line_coords(img, angle):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    straight = cv2.warpAffine(img, M, (w, h))
    if len(straight.shape)==3: gray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY)
    else: gray = straight
    
    prof = np.mean(gray[cy-20:cy+20, :], axis=0)
    peaks, _ = find_peaks(prof, height=np.mean(prof), distance=15)
    valleys, _ = find_peaks(-prof, distance=15)
    
    if len(peaks) < 2 or len(valleys) < 1: return None
    
    idx = random.randint(0, len(peaks)-2)
    x1 = peaks[idx]; x2 = peaks[idx+1]
    v_idx = np.searchsorted(valleys, x1)
    if v_idx >= len(valleys): return None
    xv = valleys[v_idx]
    
    y = random.randint(40, h-40)
    pitch = x2 - x1
    lw = (x2-x1) * 0.5; gw = (x2-x1) * 0.5
    
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    u_line = np.array([-s, c]); u_gap = np.array([c, s])
    
    M_inv = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    def rot_pt(x, y):
        p = M_inv.dot(np.array([x, y, 1]))
        return (int(p[0]), int(p[1]))
    
    return {'p1': rot_pt(x1, y), 'p2': rot_pt(x2, y), 'pv': rot_pt(xv, y),
            'pitch': pitch, 'lw': lw, 'gw': gw, 'u_line': u_line, 'u_gap': u_gap}

# ================= 7. MASTER LOOP =================

def generate_mixed_dataset():
    print(f"--- Generating {TARGET_COUNT} Mixed Metal LS Images (With LER Labels) ---")
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    if not files: return

    count = 0
    while count < TARGET_COUNT:
        fname = random.choice(files)
        img = cv2.imread(os.path.join(INPUT_FOLDER, fname))
        if img is None: continue
        
        angle = get_angle_from_filename(fname)
        labels = []
        
        # 1. LER Layer (Now with Rotation Handling)
        if random.random() < 0.30:
            # Un-rotate to vertical (0 deg) to apply LER
            img_vert, _ = rotate_image(img, -angle)
            
            # Apply Distortion
            img_dist, ler_boxes_vert = apply_ler_distortion(img_vert)
            
            # Re-rotate to original angle
            img, _ = rotate_image(img_dist, angle)
            
            # Transform LER Boxes back to rotated coordinates
            h, w = img.shape[:2]
            for cid, cx, cy, box_w, box_h in ler_boxes_vert:
                # Calculate new AABB for the rotated line segment
                ncx, ncy, nw, nh = calculate_rotated_aabb(cx, cy, box_w, box_h, angle, w, h)
                labels.append((cid, ncx, ncy, nw, nh))

        # 2. Geometric Defects
        coords = get_line_coords(img, angle)
        if coords:
            geom_roll = random.random()
            if geom_roll < 0.40: # Bridge
                if random.random() < 0.6:
                    img, bbox = warp_kissing_lines(img, coords['p1'], coords['p2'])
                else:
                    img, bbox = draw_drawn_bridge(img, coords['p1'], coords['p2'], int(coords['lw']*0.6))
                if bbox: labels.append(bbox)
            elif geom_roll < 0.80: # Open
                img, bbox = draw_open_defect(img, coords['p1'], coords['lw'], coords['gw'], coords['u_line'], coords['u_gap'])
                if bbox: labels.append(bbox)

        # 3. CMP & Particle
        if random.random() < 0.50:
            img, bbox = apply_cmp_scratch(img)
            if bbox: labels.append(bbox)
        if random.random() < 0.50:
            img, bbox = apply_particle(img)
            if bbox: labels.append(bbox)

        # 4. Finalize
        if len(labels) >= 2: 
            final_img = apply_global_grit(img)
            out_name = f"mixed_metal_{count:06d}.png"
            cv2.imwrite(f"{OUTPUT_ROOT}/Patterning Defects/images/{out_name}", final_img)
            
            with open(f"{OUTPUT_ROOT}/Patterning Defects/labels/{out_name.replace('.png','.txt')}", 'w') as f:
                for (cid, x, y, w, h) in labels:
                    nx, ny = x/TARGET_SIZE[1], y/TARGET_SIZE[0]
                    nw, nh = w/TARGET_SIZE[1], h/TARGET_SIZE[0]
                    f.write(f"{cid} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
            
            count += 1
            if count % 100 == 0: print(f"  Mixed: {count}/{TARGET_COUNT}")

if __name__ == "__main__":
    generate_mixed_dataset()
    print("Done.")