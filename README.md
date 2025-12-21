# Fall Detection via Scene Graph + Constrained Bandit 

This repo implements a fall detection pipeline using:
- GroundingDINO (zero-shot detection)
- DPT-Large (monocular depth)
- YOLOv8-Pose (human posture)
- Mask2Former (instance masks)
- SegFormer ADE20K (semantic floor carving)
- RANSAC floor plane estimation (robust floor contact)
- Scene graph generation (spatial + support relations)
- Constrained contextual bandit (LinUCB) to decide "significant change"
- Video-level fall likelihood score (0–1)

## Install

```bash
pip install -r requirements.txt
```
## Run on a video
```bash
python run_video.py --video /path/to/video.mp4 --fps_target 3 --logs_dir video_logs