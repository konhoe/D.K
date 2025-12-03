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

> Our design is inspired by CLIP-based forgery detection and adapter-style methods such as **Forensics Adapter**.  [oai_citation:0‡Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.pdf](file-service://file-6sXz1BTJznq7zy8bkD7Jqa)  

---

## 🧱 Model Architecture

- **Backbone**: CLIP visual encoder (e.g., ViT-L/14 or ViT-B/16)
- **Adapter**: D2ST adapter module
  - Takes the 2D CLIP feature map
  - Lifts it to a **(T, H, W, C)** 3D feature volume
  - Applies **spatio-temporal attention / deformable attention**
- **Input Modalities**
  - **Images**: treated as a single frame (T=1)
  - **Videos**:
    - Stage 1: sampled as **single frame** (T=1)
    - Stage 2 & 3: sampled as **12 frames** (T=12)
- **Classifier head**
  - Simple MLP / linear head applied on aggregated spatio-temporal features
  - Outputs **real vs fake** probability

---

## 🔁 3-Stage Training Pipeline

### Stage 1 – Large-scale Image & Single-frame Video Training

- **Data**
  - ~**140,000** samples
  - **Mixture of images and videos**
  - Videos are converted into **a single frame per video** (T=1)
- **Objective**
  - Make the CLIP+D2ST adapter learn **generic forgery cues** from a large-scale mixture of static images and video frames.
  - At this stage, temporal modeling is minimal; emphasis is on **spatial artifacts**.

---

### Stage 2 – Multi-frame Video-only Temporal Adaptation

- **Data**
  - ~**10,000** **video-only** samples  
  - Each video is sampled to **12 frames (T=12)**
- **Objective**
  - Adapt the adapter to capture **temporal inconsistencies** between frames:
    - Lip-sync mismatch
    - Head pose discontinuities
    - Temporal flickering / boundary jitter
  - CLIP remains frozen; D2ST and classifier are trained to exploit **spatio-temporal signals.**

---

### Stage 3 – Image–Video Alignment (Multi-frame)

- **Data**
  - ~**5,000** **mixed image & video** samples
  - Videos again sampled as **12 frames (T=12)**
- **Objective**
  - Align **image-only** and **video-based** representations in a **unified feature space**.
  - Ensure that:
    - Image features and video features are **compatible**.
    - The classifier behaves consistently across modalities.

---

## 📁 Repository Structure


- `clip_backbone.py`  
  CLIP visual backbone wrapper (e.g., ViT-L/14) and feature extraction.
- `adapter.py`  
  D2ST spatio-temporal adapter (2D→3D feature lifting, temporal attention).
- `transforms.py`  
  Make batch tensor (Each Image & Video)
- `video_utils.py`  
  Video decoding / frame sampling utilities (T=1 or T=12).
- `dataloader.py`  
  PyTorch `Dataset` / `DataLoader` definitions for images and videos.
- `processors.py`  
  Pre- / post-processing utilities (e.g., cropping, face alignment, label mapping).
  Image & video data transforms (augmentation, normalization, frame sampling).
- `cache.py`  
  Optional caching utilities for precomputed features or frame sampling metadata.
- `train.py`  
  Main training script, supports **stage 1/2/3** training.
- `export.py`  
  Script to export a trained model for inference (e.g., ONNX / TorchScript).
- `README.md`  
  You are here 🙂

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-org-or-id>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
task.ipynb 
