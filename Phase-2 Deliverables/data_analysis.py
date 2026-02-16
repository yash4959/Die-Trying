import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
import warnings
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

warnings.filterwarnings('ignore')

try:
    df = pd.read_csv("detailed_submission_scores.csv")
except FileNotFoundError:
    print("❌ ERROR: detailed_submission_scores.csv not found.")
    exit(1)

CLASSES = ['Bridge', 'Clean', 'CMP', 'Crack', 'LER', 'Open', 'Other', 'Via']
prob_cols = [f'prob_{c}' for c in CLASSES]

df['is_clean_actual'] = df['true_label'].apply(lambda x: 1 if x.lower() == 'clean' else 0)
df['is_escape'] = (df['is_clean_actual'] == 0) & (df['predicted_label'] == 'Clean')
df['entropy'] = df[prob_cols].apply(lambda row: entropy(row.values, base=2), axis=1)

sns.set_theme(style="darkgrid")
print("📊 Firing up the Final Data Science Deliverables...")

# =====================================================================
# GRAPH 1: TRUE POSITIVES VS FALSE POSITIVES (Fixed Y-Axis)
# =====================================================================
print("   ⏳ Rendering 1/4: TP vs FP Confidence Profile...")
plt.figure(figsize=(12, 6))

tp_fp_data = []
for c in CLASSES:
    tp_mask = (df['true_label'] == c) & (df['predicted_label'] == c)
    fp_mask = (df['true_label'] != c) & (df['predicted_label'] == c)
    
    tp_mean = df.loc[tp_mask, f'prob_{c}'].mean() if tp_mask.sum() > 0 else 0
    fp_mean = df.loc[fp_mask, f'prob_{c}'].mean() if fp_mask.sum() > 0 else 0
    
    tp_fp_data.append({'Class': c, 'Outcome': 'True Positive (Correct)', 'Confidence': tp_mean})
    tp_fp_data.append({'Class': c, 'Outcome': 'False Positive (Incorrect)', 'Confidence': fp_mean})

tp_fp_df = pd.DataFrame(tp_fp_data)

sns.barplot(data=tp_fp_df, x='Class', y='Confidence', hue='Outcome', palette={'True Positive (Correct)': '#2ecc71', 'False Positive (Incorrect)': '#e74c3c'})
plt.title("Model Integrity: Confidence in Correct vs. Incorrect Predictions", fontsize=14, fontweight='bold')
plt.ylabel("Average Confidence Score", fontsize=12)
plt.xlabel("Predicted Class", fontsize=12)

plt.ylim(0, 0.30) 

plt.legend(title="Prediction Outcome")
plt.tight_layout()
plt.savefig('pitch_tp_vs_fp_conf.png', dpi=300)
plt.close()

# =====================================================================
# GRAPH 2: GRANULAR ENTROPY PROFILE (The Violin Plot Fix)
# =====================================================================
print("   ⏳ Rendering 2/4: Granular Entropy Profile...")
plt.figure(figsize=(10, 8))

# Sort classes by their mean entropy so the graph is visually organized
order = df.groupby('true_label')['entropy'].mean().sort_values(ascending=False).index

sns.violinplot(
    data=df, 
    x='entropy', 
    y='true_label', 
    order=order,
    palette='magma', 
    inner='quartile', # Draws median and quartile lines inside the shapes
    cut=0 # Stops the plot from extending past the actual data min/max
)

plt.title("Sim2Real Feature Collapse: Neural Network Uncertainty by Class", fontsize=14, fontweight='bold')
plt.xlabel("Shannon Entropy (Higher = Model is more mathematically confused)", fontsize=12)
plt.ylabel("Physical Defect Class", fontsize=12)
plt.axvline(df['entropy'].mean(), color='red', linestyle='--', label=f"Global Mean Entropy: {df['entropy'].mean():.2f}")

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('pitch_entropy_profile.png', dpi=300)
plt.close()

# =====================================================================
# GRAPH 3: THE DEFECT DISTRIBUTION (The Bulletproof Legend Fix)
# =====================================================================
print("   ⏳ Rendering 3/4: Full Defect Distribution...")

defects_df = df[df['is_clean_actual'] == 0].copy()
defects_df['Status'] = defects_df['is_escape'].apply(lambda x: 'Escaped (Danger)' if x else 'Caught (Safe)')

plt.figure(figsize=(10, 5))
if not defects_df.empty:
    sns.histplot(
        data=defects_df, 
        x='prob_Clean', 
        hue='Status',
        palette={'Caught (Safe)': '#3498db', 'Escaped (Danger)': '#e74c3c'},
        bins=30, 
        kde=False,
        multiple="stack"
    )
    plt.axvline(0.125, color='black', linestyle='--', linewidth=2)
    
    plt.title("Defect Capture Dynamics: Caught vs. Escaping Defects", fontsize=14, fontweight='bold')
    plt.xlabel("Model's 'Clean' Confidence Score", fontsize=12)
    plt.ylabel("Number of Defects", fontsize=12)
    
    max_val = max(0.2, defects_df['prob_Clean'].max() + 0.02)
    plt.xlim(0, max_val)
    plt.xticks(np.arange(0, max_val + 0.02, 0.02))
    
    # 🎯 FIX: Manually build the legend patches so it physically cannot fail
    caught_patch = mpatches.Patch(color='#3498db', label='Caught (Safe)')
    escaped_patch = mpatches.Patch(color='#e74c3c', label='Escaped (Danger)')
    gate_line = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Our Bouncer Gate (0.125)')
    
    plt.legend(handles=[caught_patch, escaped_patch, gate_line], title="Defect Status", loc='upper right')
    
    plt.tight_layout()
    plt.savefig('pitch_escape_autopsy.png', dpi=300)
    plt.close()
    print("      ✅ Legend locked. Graph saved.")
else:
    print("      ⚠️ No defects found to plot.")
    
# =====================================================================
# GRAPH 4: THRESHOLD OPTIMIZATION CURVE
# =====================================================================
print("   ⏳ Rendering 4/4: Threshold Trade-off Curve...")
thresholds = np.linspace(0.0, 0.4, 100)
defect_capture_rates, clean_throughputs = [], []

for t in thresholds:
    preds = (df['prob_Clean'] >= t).astype(int) 
    tn, fp, fn, tp = confusion_matrix(df['is_clean_actual'] == 0, preds == 0, labels=[1, 0]).ravel()
    
    capture_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    throughput = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    defect_capture_rates.append(capture_rate * 100)
    clean_throughputs.append(throughput * 100)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, defect_capture_rates, label='Defect Capture Rate (Safety %)', color='#e74c3c', linewidth=3)
plt.plot(thresholds, clean_throughputs, label='Clean Throughput (Efficiency %)', color='#2ecc71', linewidth=3)

plt.axvline(0.125, color='black', linestyle='--', linewidth=2, label='Engineered Anchor Point (0.125)')
plt.scatter([0.125], [defect_capture_rates[np.abs(thresholds - 0.125).argmin()]], color='black', zorder=5, s=100)
plt.scatter([0.125], [clean_throughputs[np.abs(thresholds - 0.125).argmin()]], color='black', zorder=5, s=100)

plt.title("Industrial System Trade-off: Factory Safety vs. Edge Efficiency", fontsize=14, fontweight='bold')
plt.xlabel("Asymmetric Decision Threshold (prob_Clean)", fontsize=12)
plt.ylabel("System Performance (%)", fontsize=12)
plt.xlim(0.0, 0.3)
plt.legend(loc='center right')
plt.tight_layout()
plt.savefig('pitch_threshold_curve.png', dpi=300)
plt.close()

print("✅ SUCCESS: The pipeline is sealed. No more scripts.")