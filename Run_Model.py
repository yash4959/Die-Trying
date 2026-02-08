import os
import cv2
import numpy as np
import tensorflow as tf
import shutil

# ================= CONFIGURATION =================
MODEL_PATH = "MobileNetV2_8Class_Balanced.tflite" 
INPUT_FOLDER = "Input_Images"
OUTPUT_FOLDER = "Inspection_Results"
SAFETY_THRESHOLD = 0.50 

CLASSES = ['Bridge', 'Clean', 'CMP', 'Crack', 'LER', 'Open', 'Other', 'Via']

# ================= 1. SETUP FOLDERS =================
if os.path.exists(OUTPUT_FOLDER): shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(os.path.join(OUTPUT_FOLDER, "PASS"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "FAIL"))
os.makedirs(os.path.join(OUTPUT_FOLDER, "FLAGGED"))

print(f"🚀 Loading {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
h, w = input_details['shape'][1:3]

# Quantization Params
input_scale, input_zp = input_details['quantization']
output_scale, output_zp = output_details['quantization']

print(f"   Input Quantization: Scale={input_scale}, ZP={input_zp}")

files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"📂 Scanning {len(files)} images...")
print(f"{'FILENAME':<40} | {'STATUS':<10} | {'CONF':<8}")
print("-" * 70)

for filename in files:
    img_path = os.path.join(INPUT_FOLDER, filename)
    
    # Read Image
    stream = np.fromfile(img_path, dtype=np.uint8)
    original_img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    if original_img is None: continue

    # Resize & Grayscale
    resized = cv2.resize(original_img, (w, h))
    if resized.shape[-1] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # === FIXED PREPROCESSING ===
    # We DO NOT divide by 255 here. We let the quantization parameters handle it.
    raw_float = gray.astype(np.float32)

    # Apply Quantization: (Real_Value / Scale) + Zero_Point
    if input_scale > 0:
        input_data = (raw_float / input_scale) + input_zp
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        # Fallback for non-quantized models
        input_data = (raw_float / 255.0).astype(np.float32)

    # Add Batch Dimension [1, 224, 224, 1]
    input_data = np.expand_dims(input_data, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    # Inference
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])[0]

    # === FIXED POSTPROCESSING ===
    # Dequantize: (Quantized_Value - Zero_Point) * Scale
    if output_scale > 0:
        output_probs = (output_data.astype(np.float32) - output_zp) * output_scale
    else:
        output_probs = output_data

    # RAW DECISION (No Manual Softmax needed - TFLite outputs Probs)
    pred_idx = np.argmax(output_probs)
    confidence = output_probs[pred_idx]
    label = CLASSES[pred_idx]

    # === SAFETY & SORTING LOGIC ===
    status = "PASS"
    color = (0, 255, 0) # Green
    dest_folder = "PASS"

    if label != "Clean":
        status = f"FAIL: {label}"
        color = (0, 0, 255) # Red
        dest_folder = "FAIL"
    elif label == "Clean" and confidence < SAFETY_THRESHOLD:
        status = "FLAG: LowConf"
        color = (0, 165, 255) # Orange
        dest_folder = "FLAGGED"
    
    # Console Output
    print(f"{filename[:37]:<40} | {status:<10} | {confidence:.1%}")

    # Draw on Image
    text = f"{status} ({confidence:.1%})"
    cv2.rectangle(original_img, (0, 0), (original_img.shape[1], 40), color, -1)
    cv2.putText(original_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save
    out_path = os.path.join(OUTPUT_FOLDER, dest_folder, filename)
    is_success, buffer = cv2.imencode(".png", original_img)
    if is_success: buffer.tofile(out_path)

print("-" * 70)
print(f"✅ Done. Results sorted in '{OUTPUT_FOLDER}'")