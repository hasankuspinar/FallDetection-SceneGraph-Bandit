from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from .geometry import bbox_to_mask, iou

@dataclass
class InstanceMask:
    mask: np.ndarray
    label: str
    score: float
    box: Tuple[float, float, float, float]
    source: str

@dataclass
class SemanticMap:
    seg: np.ndarray
    id2label: Dict[int, str]

@dataclass
class MaskPack:
    instances: List[InstanceMask]
    semantic: Optional[SemanticMap]

_ADE_FLOOR = {"floor", "carpet", "rug", "ground", "mat"}
_ADE_NIGHT = {"nightstand", "cabinet", "drawer", "dresser", "chest", "side table", "end table"}

def is_floor(lbl: str) -> bool:
    L = lbl.lower()
    return any(k in L for k in ["floor", "carpet", "rug", "mat", "ground"])

def is_nightstand(lbl: str) -> bool:
    L = lbl.lower()
    return any(k in L for k in ["nightstand", "bedside", "side table", "end table"])

def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

def mask_bbox_iou(mask: np.ndarray, box_xyxy) -> float:
    if mask.sum() == 0:
        return 0.0
    return iou(mask_bbox(mask), box_xyxy)

def carve_semantic_in_box(sem: SemanticMap, box, keywords: set) -> np.ndarray:
    Hh, Ww = sem.seg.shape
    bm = bbox_to_mask(box, (Hh, Ww))
    ids = [i for i, l in sem.id2label.items() if any(k in l.lower() for k in keywords)]
    if not ids:
        return np.zeros((Hh, Ww), dtype=bool)
    return np.isin(sem.seg, np.array(ids)) & bm

def rectify_mask_in_box_with_edges(base_mask: np.ndarray, box,
                                   gray_img: np.ndarray) -> np.ndarray:
    Hh, Ww = base_mask.shape
    x0, y0, x1, y1 = map(int, box)
    x0, x1 = np.clip([x0, x1], 0, Ww)
    y0, y1 = np.clip([y0, y1], 0, Hh)
    if x1 <= x0 or y1 <= y0:
        return base_mask
    roi = gray_img[y0:y1, x0:x1]
    if roi.size == 0:
        return base_mask
    edges = cv2.Canny(roi, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return base_mask
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 64:
        return base_mask
    rect = cv2.minAreaRect(cnt)
    pts = cv2.boxPoints(rect).astype(np.int32)
    pts[:, 0] += x0
    pts[:, 1] += y0
    poly = np.zeros_like(base_mask, np.uint8)
    cv2.fillPoly(poly, [pts], 1)
    poly = poly.astype(bool)
    bm = bbox_to_mask((x0, y0, x1, y1), (Hh, Ww))
    ref = poly & bm
    return ref if ref.sum() >= 0.2 * base_mask.sum() else base_mask

def label_compatible(dino_label: str, inst_label: str) -> bool:
    d = dino_label.lower()
    i = inst_label.lower()
    pairs = [
        ("person", "person"), ("bed", "bed"), ("sofa", "sofa"), ("sofa", "couch"),
        ("couch", "couch"), ("chair", "chair"), ("bench", "bench"), ("stool", "stool"),
        ("table", "table"), ("luggage", "suitcase"), ("suitcase", "suitcase"),
        ("cabinet", "cabinet"),
    ]
    return any(a in d and b in i for a, b in pairs)

def build_instance_and_semantic_masks(
    image_pil: Image.Image,
    device: str,
    mask2former_proc,
    mask2former_model,
    segformer_proc,
    segformer_model,
    img_gray: np.ndarray,
) -> MaskPack:
    Hh, Ww = image_pil.size[1], image_pil.size[0]
    instances: List[InstanceMask] = []
    sem_map: Optional[SemanticMap] = None

    # Mask2Former
    try:
        with torch.no_grad():
            inp = mask2former_proc(images=image_pil, return_tensors="pt").to(device)
            out = mask2former_model(**inp)
        r = mask2former_proc.post_process_instance_segmentation(out, target_sizes=[(Hh, Ww)])[0]
        idmap = r["segmentation"].cpu().numpy()
        seg_infos = r["segments_info"]
        for s in seg_infos:
            if s.get("score", 0.0) < 0.40:
                continue
            m = (idmap == s["id"])
            if m.sum() == 0:
                continue
            ys, xs = np.where(m)
            x0, x1 = xs.min(), xs.max() + 1
            y0, y1 = ys.min(), ys.max() + 1
            instances.append(
                InstanceMask(
                    mask=m,
                    label=str(s.get("label", s["label_id"])),
                    score=float(s["score"]),
                    box=(float(x0), float(y0), float(x1), float(y1)),
                    source="mask2former",
                )
            )
    except Exception as e:
        print(f"[warn] Mask2Former failed: {e}")

    # SegFormer
    try:
        with torch.no_grad():
            tin = segformer_proc(images=image_pil, return_tensors="pt").to(device)
            tout = segformer_model(**tin)

        seg = segformer_proc.post_process_semantic_segmentation(
            tout, target_sizes=[(Hh, Ww)]
        )[0].cpu().numpy().astype(np.int32)

        raw = getattr(segformer_model.config, "id2label", {i: str(i) for i in range(int(seg.max()) + 1)})
        id2label: Dict[int, str] = {}
        for k, v in raw.items():
            try:
                id2label[int(k)] = str(v)
            except Exception:
                pass

        sem_map = SemanticMap(seg=seg, id2label=id2label)
    except Exception as e:
        print(f"[warn] SegFormer failed: {e}")
        sem_map = None

    return MaskPack(instances=instances, semantic=sem_map)
