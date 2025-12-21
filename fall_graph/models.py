import os
from dataclasses import dataclass
from typing import Optional
import torch
from ultralytics import YOLO
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    DPTForDepthEstimation, DPTFeatureExtractor,
    AutoImageProcessor, Mask2FormerForUniversalSegmentation,
    SegformerImageProcessor, SegformerForSemanticSegmentation,
)

@dataclass
class ModelBundle:
    device: str

    pose_model: YOLO
    tracker_model: YOLO

    dino_processor: any
    dino_model: any

    dpt_fe: any
    dpt_model: any

    mask2former_proc: any
    mask2former_model: any

    segformer_proc: any
    segformer_model: any

def pick_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_models(
    dino_model_id: str,
    dpt_model_id: str,
    mask2former_id: str,
    segformer_ade_id: str,
    yolo_pose_ckpt: str = "yolov8s-pose.pt",
    yolo_track_ckpt: str = "yolov8x.pt",
    prefer_cuda: bool = True,
) -> ModelBundle:
    device = pick_device(prefer_cuda)
    print(f"[LOADER] Using device: {device}")

    # YOLO Pose
    pose_model = YOLO(yolo_pose_ckpt)

    # YOLO tracking (BoT-SORT)
    tracker_model = YOLO(yolo_track_ckpt)

    # GroundingDINO
    dino_processor = AutoProcessor.from_pretrained(dino_model_id, trust_remote_code=True)
    dino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_model_id, trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    # DPT Depth
    dpt_fe = DPTFeatureExtractor.from_pretrained(dpt_model_id)
    dpt_model = (
        DPTForDepthEstimation.from_pretrained(dpt_model_id)
        .to(device)
        .eval()
    )

    # Mask2Former
    mask2former_proc = AutoImageProcessor.from_pretrained(mask2former_id)
    mask2former_model = (
        Mask2FormerForUniversalSegmentation.from_pretrained(mask2former_id)
        .to(device)
        .eval()
    )

    # SegFormer
    segformer_proc = SegformerImageProcessor.from_pretrained(segformer_ade_id)
    segformer_model = (
        SegformerForSemanticSegmentation.from_pretrained(segformer_ade_id)
        .to(device)
        .eval()
    )

    print("[LOADER] All core models loaded successfully! ✅")

    return ModelBundle(
        device=device,
        pose_model=pose_model,
        tracker_model=tracker_model,
        dino_processor=dino_processor,
        dino_model=dino_model,
        dpt_fe=dpt_fe,
        dpt_model=dpt_model,
        mask2former_proc=mask2former_proc,
        mask2former_model=mask2former_model,
        segformer_proc=segformer_proc,
        segformer_model=segformer_model,
    )
