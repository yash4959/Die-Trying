import os
import cv2
import numpy as np
import logging
import csv
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from sklearn.metrics import confusion_matrix, classification_report

# ================= 1. BULLETPROOF ABSOLUTE LOGGING SETUP =================
# 1. Force the absolute path. This stops Antigravity from dumping the file in the wrong folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "execution_trace.log")

# 2. Nuke any stale handlers from the IDE's kernel
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 3. Setup the dual-logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# The FileHandler now physically cannot write to the wrong directory
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(ch)

logger.info(f"🔒 LOG FILE LOCKED AND WRITING TO: {LOG_FILE_PATH}")

# ================= 2. ASSETS & CONFIG =================
MODEL_PATH = "Die-Trying_Finetuned_Model.onnx" 
TEST_DIR = "hackathon_test_dataset"
SUBMISSION_CSV = "detailed_submission_scores.csv"
REPORT_TXT = "classification_metrics.txt"
IMG_SIZE = 224

CLASSES = ['Bridge', 'Clean', 'CMP', 'Crack', 'LER', 'Open', 'Other', 'Via']
# Explicitly forcing 'Via' to index 7 to isolate it in matrices
FOLDER_TO_CLASS_IDX = {'bridge': 0, 'clean': 1, 'cmp': 2, 'crack': 3, 'ler': 4, 'open': 5, 'other': 6, 'particle': 6, 'via': 7}

# --- ⚡ PHASE-3 COMPLIANT STATIC CONSTANTS ---
GLOBAL_MEAN = 108.02 
GLOBAL_STD = 42.94   

# --- ⚔️ THE ASYMMETRIC MCU RISK GATE ---
GLOBAL_CLEAN_THRESHOLD = 0.125 
OTHER_SUSPICION_THRESHOLD = 0.120 

# ================= 3. BARE-METAL INFERENCE PIPELINE =================
# ================= 3. BARE-METAL INFERENCE PIPELINE (PROFILED) =================
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    logger.info("✅ ONNX Model loaded successfully into CPUExecutionProvider.")
except Exception as e:
    logger.error(f"❌ Failed to load ONNX: {e}")
    exit(1)

predictions, y_true, y_pred = [], [], []

# --- 📊 START HARDWARE PROFILER ---
process = psutil.Process(os.getpid())
baseline_ram = process.memory_info().rss / (1024 * 1024) # Convert bytes to MB
total_inference_time_ms = 0.0

logger.info(f"🚀 Initiating Hardware-Compliant Inference Sequence...")
logger.info(f"⚙️ Static MCU Scaling: MEAN={GLOBAL_MEAN}, STD={GLOBAL_STD}")
logger.info(f"🛡️ Active Risk Gate: Clean_Thresh={GLOBAL_CLEAN_THRESHOLD}, Other_Veto={OTHER_SUSPICION_THRESHOLD}")
logger.info("-------------------------------------------------------------------------------")
logger.info(f"{'FILENAME':<20} | {'TRUE':<8} | {'PRED':<8} | {'CONFIDENCE':<12} | {'LATENCY (ms)'}")
logger.info("-------------------------------------------------------------------------------")

for folder in os.listdir(TEST_DIR):
    f_path = os.path.join(TEST_DIR, folder)
    if not os.path.isdir(f_path): continue
    
    true_class_name = folder.lower()
    true_idx = FOLDER_TO_CLASS_IDX.get(true_class_name, 6) 

    for file in os.listdir(f_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
        # Step A & B: Image Load and Static Scale
        img_raw = cv2.resize(cv2.imread(os.path.join(f_path, file), cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        img_norm = (img_raw - GLOBAL_MEAN) / (GLOBAL_STD + 1e-5)
        img_scaled = np.clip((img_norm * 50.0) + 128.0, 0, 255).astype(np.float32)
        batch = np.expand_dims(np.expand_dims(img_scaled, axis=0), axis=-1)
        
        # ⏱️ START LATENCY TIMER (Isolating strictly the ONNX math)
        t_start = time.perf_counter()
        
        # Step C: Raw Predict
        raw_out = session.run(None, {input_name: batch})[0][0]
        
        # ⏱️ STOP LATENCY TIMER
        t_end = time.perf_counter()
        infer_time_ms = (t_end - t_start) * 1000
        total_inference_time_ms += infer_time_ms
        
        exp_scores = np.exp(raw_out - np.max(raw_out))
        probs = exp_scores / exp_scores.sum()
        formatted_probs = [round(float(p), 4) for p in probs]
        
        clean_conf = probs[1]
        other_conf = probs[6]
        base_pred = np.argmax(probs)
        
        # Step D: The Asymmetric Bouncer
        if clean_conf >= GLOBAL_CLEAN_THRESHOLD or base_pred == 6:
            if other_conf < OTHER_SUSPICION_THRESHOLD:
                if clean_conf >= GLOBAL_CLEAN_THRESHOLD:
                    final_idx = 1
                else:
                    final_idx = 6
            else:
                final_idx = 6
        else:
            probs[1] = 0.0
            final_idx = np.argmax(probs)
        
        # 📝 LOG THE INDIVIDUAL TELEMETRY
        logger.info(f"{file:<20} | {CLASSES[true_idx]:<8} | {CLASSES[final_idx]:<8} | {clean_conf:<12.4f} | {infer_time_ms:.2f} ms")
        
        row_payload = [file, CLASSES[true_idx], CLASSES[final_idx], final_idx] + formatted_probs
        predictions.append(row_payload)
        y_true.append(true_idx)
        y_pred.append(final_idx)

# --- 📊 STOP HARDWARE PROFILER ---
peak_ram = process.memory_info().rss / (1024 * 1024)
ram_overhead = peak_ram - baseline_ram
avg_latency = total_inference_time_ms / len(y_true)

logger.info("-------------------------------------------------------------------------------")
logger.info("⏱️ PHASE 3 DEPLOYMENT TELEMETRY REPORT:")
logger.info(f"   ▶ Total Images Processed : {len(y_true)}")
logger.info(f"   ▶ Total Inference Time   : {total_inference_time_ms:.2f} ms")
logger.info(f"   ▶ Average Latency / Image: {avg_latency:.2f} ms")
logger.info(f"   ▶ Process RAM Footprint  : {peak_ram:.2f} MB (+{ram_overhead:.2f} MB during execution)")
logger.info("-------------------------------------------------------------------------------")

# ================= 4. GENERATING AUDIT LOGS & VISUALS =================

logger.info(f"💾 Writing cleanly formatted softmax confidence scores to {SUBMISSION_CSV}...")
csv_headers = ['filename', 'true_label', 'predicted_label', 'predicted_idx'] + [f'prob_{c}' for c in CLASSES]
with open(SUBMISSION_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(predictions)

y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)

# Forced Matrix Ordering: Via(7) first, Clean(1) last
custom_order_idx = [7, 0, 2, 3, 4, 5, 6, 1]
custom_order_names = [CLASSES[i] for i in custom_order_idx]

report = classification_report(y_true_arr, y_pred_arr, labels=custom_order_idx, target_names=custom_order_names, zero_division=0)
with open(REPORT_TXT, 'w') as f:
    f.write("=======================================================\n")
    f.write("PHASE 3 BARE-METAL COMPLIANT VERDICT\n")
    f.write(f"STATIC MCU SCALING: MEAN={GLOBAL_MEAN}, STD={GLOBAL_STD}\n")
    f.write("=======================================================\n\n")
    f.write(report)
logger.info(f"📄 Text metrics saved to {REPORT_TXT}")

logger.info("🎨 Rendering 8-Class Matrix (Greens)...")
cm_8 = confusion_matrix(y_true_arr, y_pred_arr, labels=custom_order_idx)
plt.figure(figsize=(11, 8))
sns.heatmap(cm_8, annot=True, fmt='d', cmap='Greens', xticklabels=custom_order_names, yticklabels=custom_order_names)
plt.title("8-Class Bare-Metal Defect Classification")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig('8_class_matrix.png', dpi=300)
plt.close()

logger.info("🎨 Rendering Industrial Binary Matrix...")
y_true_bin = (y_true_arr == 1).astype(int)
y_pred_bin = (y_pred_arr == 1).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()

cm_binary = np.array([[tn, fp], [fn, tp]])
binary_labels = ['Defect', 'Clean']

plt.figure(figsize=(6, 5))
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Reds', xticklabels=binary_labels, yticklabels=binary_labels)
plt.title("Industrial Safety Verdict (Binary)")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig('binary_safety_matrix.png', dpi=300)
plt.close()

logger.info("\n=======================================================")
logger.info(f" 🛡️ FINAL BINARY VERDICT | Correct Cleans: {tp} | Total Escapes: {fp}")
logger.info("=======================================================")
logger.info("🏁 Execution Terminated. All deliverables packaged and saved.")