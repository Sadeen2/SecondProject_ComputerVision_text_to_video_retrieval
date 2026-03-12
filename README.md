# README.md

# Text-to-Video Retrieval (MSVD)

#  Method: Dual-Encoder + Transfer Learning + Contrastive Learning + Cached Video Index

## 📌 Project Overview
This project implements a **Text-to-Video Retrieval system** that retrieves the most relevant video given a natural language query. The system is evaluated on the **MSVD (Microsoft Video Description) dataset**, a standard benchmark for video–text retrieval tasks.

The proposed approach adopts a **dual-encoder architecture**, where:
- Videos are encoded using a **CNN-based visual encoder**
- Text queries are encoded using a **Transformer-based language encoder**
- Both modalities are projected into a **shared embedding space** for similarity-based retrieval

---

## 🧠 Model Architecture

### 1. Video Encoder
- Backbone: **ResNet18**
- Input: RGB video frames
- Number of frames per video: **4**
- Image resolution: **224 × 224**
- Frame features are aggregated to represent each video

### 2. Text Encoder
- Backbone: **DistilBERT (distilbert-base-uncased)**
- Input: Natural language captions
- Lightweight transformer optimized for efficiency
- Outputs sentence-level embeddings

### 3. Joint Embedding Space
- Both video and text embeddings are projected into a common latent space
- Retrieval is performed using **cosine similarity**
- Contrastive learning objective is used during training

---

## 📂 Dataset

### MSVD (Microsoft Video Description Dataset)
- ~2,000 short video clips
- ~40 textual captions per video
- Widely used for video–text retrieval research

Dataset split:
- Training
- Validation
- Testing

---

Pipeline:
MSVD → Frame Extraction → CNN Feature Extraction → Temporal Modeling → Text Transformer
→ Joint Embedding → Contrastive Training → Cached Index → Top-K Retrieval

---
## ⚙️ Implementation Details

- Programming Language: **Python**
- Framework: **PyTorch**
- GPU Support: **CUDA**
- Batch size: **32**
- Number of frames: **4**
- Image size: **224**
- Checkpoint format: `.pt`

---

## 0) VS Code Setup (Python 3.11)

### A) Create virtual environment

Open VS Code terminal in the project root:

```bat
py -3.11 -m venv .venv
```

### B) Activate environment

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```bat
.\.venv\Scripts\activate.bat
```

### C) Select interpreter in VS Code

* Press `Ctrl+Shift+P`
* Select: **Python: Select Interpreter**
* Choose: `.venv`

### D) Install dependencies

```bat
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

---

## 1) Place MSVD Dataset Files

Put MSVD files here:

```
data/raw/
├── videos/              # YouTubeClips (all video files)
└── annotations.txt      # MSVD captions file
```

---

## 2) Run Order (Commands)

### Step 1 — Extract frames

```bat
py data\extract_frames.py --num_frames 8
```

Output:

* `data/frames/<video_id>/000.jpg ...`

### Step 1 — Build cleaned annotations + splits

```bat
py data\build_annotations.py
```

Output:

* `data/processed/annotations.csv` with columns: `video_id,caption,split`

---

## 3) Train (Contrastive Learning)

```bat
!python -m training.train \
  --device cuda \
  --epochs 5 \
  --batch_size 32 \
  --backbone resnet18 \
  --freeze_backbone \
  --num_frames 4
```
on Google Colab 

Outputs:

* `checkpoints/model_last.pt`
* `checkpoints/model_best.pt`

---

## 4) Evaluate (Retrieval Metrics)

```bat
py training\validate.py --checkpoint checkpoints\model_best.pt
```

Reports:

* Recall@K (R@1, R@5, R@10)
* MedR, MeanR, mAP
*  retrieval time if implemented

---

## 5) Build Cached Video Index (Fast Retrieval)

```bat
py retrieval\build_video_index.py --checkpoint checkpoints\model_best.pt
```

Outputs:

* `index/video_embeddings.npy`
* `index/video_ids.json`

---

## 6) Search (Top-K Retrieval)

```bat
py retrieval\search.py --query "a man is playing guitar" --top_k 5
```

---
Run Retrieval and Validation
```
py main.py

```
The script supports:

Interactive text-to-video search

Visualization of retrieved video frames

Optional validation evaluation at the end

## Notes

* Use `py` everywhere (Python launcher on Windows).
* If you get CUDA/GPU issues, PyTorch will still run on CPU (slower but works).

