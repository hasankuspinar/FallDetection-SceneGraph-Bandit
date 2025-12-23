# Graph-First Fall Detection with Scene Graphs and Constrained Bandits

This repository implements a **graph-first, video-based fall detection system** that models each video frame as a **scene graph** and detects falls by analyzing **relational changes over time** using a **constrained contextual bandit (LinUCB)**.

Instead of relying on single-frame classification or end-to-end black-box models, the system focuses on **explainable structural cues** such as posture transitions, floor contact, and support-object changes. This design is particularly suitable for **safety-critical elderly monitoring scenarios**, where minimizing missed falls (false negatives) is essential.

---

## Key Contributions

- Scene graph–based representation of each video frame  
- Robust floor estimation using monocular depth + RANSAC  
- Human posture reasoning (standing, sitting, lying, falling)  
- Person–support relationship modeling (chair, bed, floor)  
- Temporal change analysis between consecutive scene graphs  
- Constrained LinUCB bandit for adaptive, low-noise alarm decisions  
- Interpretable alarms with human-readable explanations  
- Video-level fall likelihood score in the range **[0, 1]**

---

## System Overview

Each frame in the video is processed as follows:

1. Perception & Geometry  
2. Scene Graph Construction  
3. Graph Delta Extraction  
4. Constrained Bandit Decision  
5. Video-Level Fall Scoring  

A fall is characterized not by a single cue, but by a **structural signature** such as:

standing_on → falling_towards_floor → lying_on_floor

---

## Models Used

| Component | Model |
|---------|------|
| Zero-shot object detection | GroundingDINO |
| Monocular depth estimation | DPT-Large |
| Human pose estimation | YOLOv8-Pose |
| Instance segmentation | Mask2Former |
| Semantic segmentation (floor) | SegFormer (ADE20K) |
| Floor plane estimation | RANSAC |
| Decision mechanism | Constrained LinUCB |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run on a Video

```bash
python run_video.py --video /path/to/video.mp4 --fps_target 3 --logs_dir video_logs
```

### Outputs
- Scene graph JSON files  
- Frame-level bandit decisions  
- Interpretable alarm explanations  
- Final video-level fall probability score  

---

## Datasets Used for Evaluation

- **URFall**
- **CAUCAFall**

---

## Experimental Results

The proposed method was evaluated on **URFall** and **CAUCAFall** datasets and compared against recent vision-language and video-based baselines.

### Quantitative Results

| Method | Precision | Recall | F1-score | Accuracy |
|------|----------|--------|----------|----------|
| InstructBLIP (baseline) | 0.471 | 1.000 | 0.640 | 0.471 |
| SmolVLM2-500M | 0.824 | 0.735 | 0.710 | 0.735 |
| Video-LLaVA-7B | 0.784 | 0.783 | 0.783 | 0.782 |
| Qwen3-VL-8B-Instruct | 0.929 | 0.924 | 0.924 | 0.924 |
| **Ours (Scene Graph + LinUCB)** | **0.686** | **0.960** | **0.800** | **0.760** |

### Discussion

- The proposed method achieves **very high recall (0.96)**, significantly reducing **false negatives**, which is critical for fall detection systems.
- While precision is lower than large vision-language models, the system produces **fewer missed falls** and more **interpretable decisions**.
- Compared to end-to-end video transformers, the method uses **lighter-weight models** and explicit relational reasoning.
- Scene graph–based reasoning enables robustness against common false positives caused by beds, chairs, and lying poses.

---


