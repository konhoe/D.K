# D2ST-Adapter for Generalizable Deepfake Detection

> **TL;DR**  
> We adapt CLIP with a D2ST-based spatio-temporal adapter to detect face forgeries from both images and videos.  
> A 3-stage training pipeline (image+single-frame, multi-frame video only, and image–video alignment) allows the model to generalize across datasets and manipulation types.

---

## 🔍 Overview

This repository implements a **CLIP-based deepfake detector** with a **D2ST (Deformable-Depth Spatio-Temporal) adapter** on top of the CLIP visual backbone.

- We take the **2D feature map** from a CLIP image encoder and **reconstruct it into a 3D feature volume**.
- The D2ST adapter injects:
  - **Spatial cues** (for local forgery artifacts),
  - **Temporal cues** (frame-to-frame inconsistencies),
  - While keeping CLIP itself **frozen** for better generalization.

Our training is divided into **three stages**:

1. **Stage 1 – Large-scale image & single-frame video pretraining**
2. **Stage 2 – Multi-frame video-only temporal adaptation**
3. **Stage 3 – Image–video alignment with multi-frame input**

The overall goal is to obtain a **robust, generalizable, and computation-friendly** deepfake detector.

> Our design is inspired by CLIP-based forgery detection and adapter-style methods such as **Forensics Adapter**  
> (“Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection”, CVPR 2025).

---

## 📋 Data Structure

- **High-level structure of datasets & splits**

<img width="253" height="205" alt="image" src="https://github.com/user-attachments/assets/42f2cae9-2c11-4df5-ac5b-6a27b62b1eda" />
<img width="275" height="205" alt="image" src="https://github.com/user-attachments/assets/001a66c3-ae10-44e7-9e98-1b6dc3768a49" />


---

## 🧱 Model Architecture

- **Backbone**: CLIP visual encoder (e.g., ViT-L/14 or ViT-B/16)
- **Adapter**: D2ST adapter module
  - Takes the 2D CLIP feature map
  - Lifts it to a **(T, H, W, C)** 3D feature volume
  - Applies **spatio-temporal attention / deformable attention**
- **Input Modalities**
  - **Images**: treated as a single frame (**T = 1**)
  - **Videos**:
    - Stage 1: sampled as **single frame** (**T = 1**)
    - Stage 2 & 3: sampled as **12 frames** (**T = 12**)
- **Classifier head**
  - Simple MLP / linear head applied on aggregated spatio-temporal features
  - Outputs **real vs fake** probability

---

## 🔁 3-Stage Training Pipeline

### Stage 1 – Large-scale Image & Single-frame Video Training

- **Data**
  - ~**140,000** samples
  - **Mixture of images and videos**
  - Videos are converted into **a single frame per video** (T = 1)
- **Objective**
  - Make the CLIP + D2ST adapter learn **generic forgery cues** from a large-scale mixture of static images and video frames.
  - Temporal modeling is minimal; emphasis is on **spatial artifacts**.

---

### Stage 2 – Multi-frame Video-only Temporal Adaptation

- **Data**
  - ~**10,000** **video-only** samples  
  - Each video is sampled to **12 frames (T = 12)**
- **Objective**
  - Adapt the adapter to capture **temporal inconsistencies** between frames:
    - Lip-sync mismatch
    - Head pose discontinuities
    - Temporal flickering / boundary jitter
  - CLIP remains frozen; D2ST and classifier are trained to exploit **spatio-temporal signals**.

---

### Stage 3 – Image–Video Alignment (Multi-frame)

- **Data**
  - ~**5,000** **mixed image & video** samples
  - Videos again sampled as **12 frames (T = 12)**
- **Objective**
  - Align **image-only** and **video-based** representations in a **unified feature space**.
  - Ensure that:
    - Image features and video features are **compatible**.
    - The classifier behaves consistently across modalities.

---

## 🧪 Additional Experiment: Paired Image–Video Training

Apart from the 3-stage pipeline above, we also conducted a **separate experiment** using a **paired image–video dataset**.

- **Dataset**
  - ~**5,000** pairs, where each pair consists of:
    - one still face **image**, and  
    - its corresponding **video clip** (sampled to 12 frames, **T = 12**)
- **Training Setup**
  - The model was trained **as an independent experiment**,  
    not as part of Stage 3 but with a **different training schedule**.
  - Each mini-batch always contained **paired image–video examples**,  
    encouraging explicit image–video consistency.
- **Observation**
  - This paired training **did not provide a critical performance improvement** compared to the main 3-stage pipeline.
  - We observed only **marginal gains** in some cross-modality consistency cases,  
    so this experiment is **not required** to reproduce our main results.

---

## 📁 Repository Structure

- `clip_backbone.py`  
  CLIP visual backbone wrapper (e.g., ViT-L/14) and feature extraction.
- `adapter.py`  
  D2ST spatio-temporal adapter (2D → 3D feature lifting, temporal attention).
- `transforms.py`  
  Make batch tensor (each image & video).
- `video_utils.py`  
  Video decoding / frame sampling utilities (`T = 1` or `T = 12`).
- `dataloader.py`  
  PyTorch `Dataset` / `DataLoader` definitions for images and videos.
- `processors.py`  
  Pre- / post-processing utilities (e.g., cropping, face alignment, label mapping).  
  Image & video data transforms (augmentation, normalization, frame sampling).
- `cache.py`  
  Optional caching utilities for precomputed features or frame sampling metadata.
- `train.py`  
  Main training script, supports **stage 1 / 2 / 3** training (and `paired` / `eval` modes if configured).
- `export.py`  
  Script to export a trained model for inference (e.g., ONNX / TorchScript).
- `task.ipynb`  
  Example notebook showing end-to-end training / evaluation.
- `README.md`  
  You are here 🙂

---

## 🖨️ Result & Observations

We qualitatively evaluated the model on:
	1.	Unedited real images of ordinary people
	2.	Edited / retouched images, such as:
	•	celebrity portraits,
	•	images with heavy beauty filters,
	•	skin smoothing, reshaping, or strong color/contrast edits
	3.	Additional public deepfake / forgery datasets (images & videos)

Key observations:
	•	On non-edited real images (daily photos without strong post-processing),
the model detects “real” very well and rarely misclassifies them as fake.
	•	On edited / beautified images (e.g., celebrity photos, heavy filters, retouching),
the model tends to:
	•	confuse them with fakes, or
	•	assign a high fake probability, even though they are real but aesthetically processed.
	•	On standard deepfake datasets, the model behaves as a reasonable CLIP-based baseline with temporal modeling,
but the tendency to treat “over-edited real faces” as suspicious remains a systematic bias.

---
## ⚙️ Installation

```bash
git clone https://github.com/<your-org-or-id>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
task.ipynb
