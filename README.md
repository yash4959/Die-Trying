# 🛡️ Edge AI-Based Defect Classification System
### Target: NXP i.MX RT1170 Crossover MCU

![NXP](https://img.shields.io/badge/Platform-NXP%20i.MX%20RT1170-orange)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-green)
![Team](https://img.shields.io/badge/Team-Pandit%20Deendayal%20Energy%20University-blue)

**A high-throughput, physics-aware defect classification engine designed to bring Industry 4.0 intelligence directly to the semiconductor edge.**

---

## 👥 The Team
[cite_start]**College:** Pandit Deendayal Energy University [cite: 20]

| Role | Name | Academic Year |
| :--- | :--- | :--- |
| **Team Leader** | Aditya Malpani | [cite_start]3rd Year B.Tech [cite: 17] |
| **Member** | Yash Dobariya | [cite_start]3rd Year B.Tech [cite: 17] |
| **Member** | Krins Italiya | [cite_start]3rd Year B.Tech [cite: 17] |
| **Member** | Pratyush Maheshwari | [cite_start]3rd Year B.Tech [cite: 17] |

---

## 🚩 The Problem: The "Data Bottleneck"
Modern semiconductor fabs generate massive volumes of die inspection images (AOI/SEM). Centralized manual or cloud-based inspection pipelines suffer from:
1.  [cite_start]**High Latency:** Transmitting gigabytes of raw image data creates unacceptable delays[cite: 32].
2.  [cite_start]**Scalability limits:** Centralized servers become bottlenecks as production volume increases[cite: 32].
3.  [cite_start]**Data Scarcity:** Real-world defect data is proprietary and highly imbalanced, making it difficult to train robust AI models[cite: 30, 31].

[cite_start]**The Need:** A real-time, on-device classification system that filters defects *at the source*[cite: 34].

---

## 💡 Our Solution
We have built an end-to-end **Edge AI Pipeline** that solves both the data scarcity problem and the deployment constraint.

### 1. The "Data Factory" (Synthetic Engine)
Instead of relying on scarce public datasets, we built a **Physics-Aware Synthetic Data Engine** that generates photorealistic SEM images.
* [cite_start]**Innovation:** Simulates physical manufacturing defects (Resist Collapse, Etch Failure, Shorts) rather than just pasting white boxes[cite: 61].
* [cite_start]**Outcome:** 22,000+ labeled images with "pixel-perfect" ground truth[cite: 65].

### 2. The "Edge Brain" (MobileNetV2)
We utilize a highly optimized **MobileNetV2 (0.75α)** architecture tailored for the NXP i.MX RT1170.
* [cite_start]**Strategy:** Constraint-driven design where accuracy, latency, and memory are co-optimized from day one[cite: 40, 53].
* **Performance:** Achieves **96% Accuracy** on synthetic baselines and **92% Defect Detection Rate** on real-world hardware tests.

---

## 📂 Repository Structure

This repository is organized into the modular components of our pipeline:

### 🏭 [`/data_pipeline`](./data_pipeline)
**The Physics-Aware Synthetic Data Engine.**
* Scripts for generating 8 classes of semiconductor defects (Bridge, Open, CMP, etc.).
* Domain adaptation logic (SEM noise, charging effects, focus blur).
* *Key Tech: Python, OpenCV, NumPy.*

### 🧠 [`/model_training`](./model_training)
**The Model Development & Quantization Workflow.**
* Jupyter notebooks for Transfer Learning and Fine-Tuning.
* Implementation of **Focal Loss** for hard-mining rare defects.
* **TFLite Conversion** scripts for Int8 quantization (1.7MB final size).
* *Key Tech: TensorFlow, Keras, TFLite.*

---

## 🚀 Key Innovations

* [cite_start]**Bridging the Data Gap:** We created a proprietary dataset capturing realistic defects and inspection conditions unavailable in public datasets[cite: 61].
* [cite_start]**Data-First Strategy:** Our model performance is driven by domain-specific data curation and "Sim-to-Real" transfer learning, not just generic vision features[cite: 62].
* [cite_start]**Deployment-Ready:** Output is structured for immediate integration with Industry 4.0 monitoring systems (Defect Type + Confidence Score)[cite: 45].

---

## 📉 Impact & Feasibility

* [cite_start]**Accelerated Decisions:** Reduces the delay between detection and corrective action[cite: 73].
* [cite_start]**Yield Stabilization:** Prevents defective dies from moving downstream in the process[cite: 74].
* [cite_start]**Infrastructure Savings:** Reduces the need for massive bandwidth to transmit raw images[cite: 78].

---

## 🛠️ Technology Stack
* **Language:** Python 3.9+
* **Frameworks:** TensorFlow 2.x, OpenCV
* **Target Hardware:** NXP i.MX RT1170 (Crossover MCU)
* **Optimization:** TFLite Micro (Int8 Quantization)

---

*Submitted for the IESA DeepTech Hackathon 2026.*