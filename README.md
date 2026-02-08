# 🛡️ Edge AI-Based Defect Classification System
### Target: NXP i.MX RT1170 Crossover MCU

![NXP](https://img.shields.io/badge/Platform-NXP%20i.MX%20RT1170-orange)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-green)
![Team](https://img.shields.io/badge/Team-Pandit%20Deendayal%20Energy%20University-blue)

**A high-throughput, physics-aware defect classification engine designed to bring Industry 4.0 intelligence directly to the semiconductor edge.**

---

## 👥 The Team
**College:** Pandit Deendayal Energy University 

| Role | Name | Academic Year |
| :--- | :--- | :--- |
| **Team Leader** | Aditya Malpani | 3rd Year B.Tech  |
| **Member** | Yash Dobariya | 3rd Year B.Tech |
| **Member** | Krins Italiya | 3rd Year B.Tech  |
| **Member** | Pratyush Maheshwari | 3rd Year B.Tech |

---

## 🚩 The Problem: The "Data Bottleneck"
Modern semiconductor fabs generate massive volumes of die inspection images (AOI/SEM). Centralized manual or cloud-based inspection pipelines suffer from:
1.  **High Latency:** Transmitting gigabytes of raw image data creates unacceptable delays.
2.  **Scalability limits:** Centralized servers become bottlenecks as production volume increases.
3.  **Data Scarcity:** Real-world defect data is proprietary and highly imbalanced, making it difficult to train robust AI models.

**The Need:** A real-time, on-device classification system that filters defects *at the source*.

---

## 💡 Our Solution
We have built an end-to-end **Edge AI Pipeline** that solves both the data scarcity problem and the deployment constraint.

### 1. The "Data Factory" (Synthetic Engine)
Instead of relying on scarce public datasets, we built a **Physics-Aware Synthetic Data Engine** that generates photorealistic SEM images.
* **Innovation:** Simulates physical manufacturing defects (Resist Collapse, Etch Failure, Shorts) rather than just pasting white boxes.
* **Outcome:** 22,000+ labeled images with "pixel-perfect" ground truth.

### 2. The "Edge Brain" (MobileNetV2)
We utilize a highly optimized **MobileNetV2 (0.75α)** architecture tailored for the NXP i.MX RT1170.
* **Strategy:** Constraint-driven design where accuracy, latency, and memory are co-optimized from day one.
* **Performance:** Achieves **96% Accuracy** on synthetic baselines and **92% Defect Detection Rate** on real-world hardware tests.

---

## 📂 Repository Structure

This repository is organized into the modular components of our pipeline:

### 🏭 [`/data_pipeline`](./Synthetic%20Data%20Pipeline)
**The Physics-Aware Synthetic Data Engine.**
* Scripts for generating 8 classes of semiconductor defects (Bridge, Open, CMP, etc.).
* Domain adaptation logic (SEM noise, charging effects, focus blur).
* *Key Tech: Python, OpenCV, NumPy.*

### 🧠 [`/model_training`](./ML%20Model%20Notebooks)
**The Model Development & Quantization Workflow.**
* Jupyter notebooks for Transfer Learning and Fine-Tuning.
* Implementation of **Focal Loss** for hard-mining rare defects.
* **TFLite Conversion** scripts for Int8 quantization (1.7MB final size).
* *Key Tech: TensorFlow, Keras, TFLite.*

---

## 🚀 Key Innovations

* **Bridging the Data Gap:** We created a proprietary dataset capturing realistic defects and inspection conditions unavailable in public datasets.
* **Data-First Strategy:** Our model performance is driven by domain-specific data curation and "Sim-to-Real" transfer learning, not just generic vision features.
* **Deployment-Ready:** Output is structured for immediate integration with Industry 4.0 monitoring systems (Defect Type + Confidence Score).

---

## 📉 Impact & Feasibility

* **Accelerated Decisions:** Reduces the delay between detection and corrective action.
* **Yield Stabilization:** Prevents defective dies from moving downstream in the process.
* **Infrastructure Savings:** Reduces the need for massive bandwidth to transmit raw images.

---

## 🛠️ Technology Stack
* **Language:** Python 3.9+
* **Frameworks:** TensorFlow 2.x, OpenCV
* **Target Hardware:** NXP i.MX RT1170 (Crossover MCU)
* **Optimization:** TFLite Micro (Int8 Quantization)

---


*Submitted for the IESA DeepTech Hackathon 2026.*




