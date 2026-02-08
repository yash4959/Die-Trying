import os
import tensorflow as tf
import tf2onnx
from tensorflow.keras import layers

# ================= CONFIGURATION =================
# Ensure this matches your file name exactly
MODEL_NAME = "best_model.keras" 
ONNX_NAME = "TeamName_Phase1_Baseline.onnx"

# ================= LOAD MODEL =================
print(f"🧠 Loading local model: {MODEL_NAME}...")

if not os.path.exists(MODEL_NAME):
    raise FileNotFoundError(f"❌ Error: Could not find '{MODEL_NAME}' in this folder.")

# Safety definitions for Custom Objects
# (Required to load the file even if you aren't using the layers right now)
@tf.keras.utils.register_keras_serializable()
class SobelLayer(layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, inputs): return inputs 
    def get_config(self): return super().get_config()

def focal_loss_fixed(y_true, y_pred): return 0

try:
    model = tf.keras.models.load_model(
        MODEL_NAME, 
        custom_objects={'SobelLayer': SobelLayer, 'focal_loss_fixed': focal_loss_fixed}
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# ================= CONVERT TO ONNX =================
print(f"🔄 Converting to {ONNX_NAME}...")

# Define Input Signature (Batch, 224, 224, 1 Channel)
spec = (tf.TensorSpec((None, 224, 224, 1), tf.float32, name="input"),)

# Convert
output_path = ONNX_NAME
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)

print(f"✅ SUCCESS! Saved to: {output_path}")
print(f"📦 File Size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")