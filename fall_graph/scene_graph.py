import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RANSACRegressor

from .geometry import bbox_to_mask, centroid, iou
from .masks import (
    MaskPack, is_floor, is_nightstand,
    build_instance_and_semantic_masks,
    carve_semantic_in_box, rectify_mask_in_box_with_edges,
    label_compatible, mask_bbox_iou
)
from .pose import classify_posture_from_kpts, yolo_pose_people
from .tracking import get_tracked

# ---------------- Data types ----------------
@dataclass
class AlignedDet:
    label: str
    score: float
    box: Tuple[float, float, float, float]
    mask: np.ndarray
    source: str
    depth_median: float

# ---------------- State (persistent across frames) ----------------
class SceneGraphState:
    def __init__(self):
        # permanent IDs
        self.eternal_floor_id = None
        self.eternal_bed_id = None
        self.eternal_sofa_id = None
        self.person_id_map: Dict[int, str] = {}
        self.next_person_id = 1

        # temporal smoothing
        self.prev_depth_map: Optional[np.ndarray] = None
        self.prev_floor_params: Optional[Tuple[float, float, float]] = None

        self.posture_history: Dict[str, deque] = {}
        self.support_history: Dict[str, deque] = {}
        self.rel_history: Dict[Tuple[str, str], deque] = {}

        self.floor_params: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.floor_mask_global: Optional[np.ndarray] = None

# ---------------- RANSAC floor plane ----------------
def ransac_floor_plane(depth_map: np.ndarray, floor_mask: np.ndarray,
                      img_wh: Tuple[int, int], depth_rng: float) -> Tuple[float, float, float]:
    Ww, Hh = img_wh
    ys, xs = np.where(floor_mask)
    if xs.size < 50:
        return 0.0, 0.0, 0.0

    X = (xs / Ww - 0.5) * 2.0
    Y = (ys / Hh - 0.5) * 2.0
    Z = depth_map[ys, xs]
    data = np.vstack([X, Y]).T

    try:
        ransac = RANSACRegressor(
            min_samples=10,
            residual_threshold=0.015 * depth_rng,
            max_trials=500
        )
        ransac.fit(data, Z)
        A, B = ransac.estimator_.coef_
        C = ransac.estimator_.intercept_
        return float(A), float(B), float(C)
    except Exception:
        return 0.0, 0.0, 0.0

def depth_diff_to_plane(x: float, y: float, depth: float,
                        params: Tuple[float, float, float],
                        img_wh: Tuple[int, int],
                        closer_is_smaller: bool) -> float:
    Ww, Hh = img_wh
    A, B, C = params
    X_norm = (x / Ww - 0.5) * 2.0
    Y_norm = (y / Hh - 0.5) * 2.0
    Z_plane = A * X_norm + B * Y_norm + C
    diff = depth - Z_plane
    return (-diff) if closer_is_smaller else diff

# ---------------- Helpers ----------------
def dilate_bool(m, k=3):
    ker = np.ones((k, k), np.uint8)
    return cv2.dilate(m.astype(np.uint8), ker, 1) > 0

def depth_slice(m, z, q_lo=0.25, q_hi=0.75):
    vals = z[m]
    if vals.size == 0:
        return None
    return float(np.quantile(vals, q_lo)), float(np.quantile(vals, q_hi))

def overlap_in_depth(a, b, z, tol):
    za = depth_slice(a, z)
    zb = depth_slice(b, z)
    if za is None or zb is None:
        return False
    inter = max(0.0, min(za[1], zb[1]) - max(za[0], zb[0]))
    union = max(za[1], zb[1]) - min(za[0], zb[0]) + 1e-6
    return (inter / union) >= tol

def underlap_ratio(person_mask, support_mask, y0, y1, pad=6):
    Hh, Ww = person_mask.shape
    y0 = max(0, min(Hh, y0))
    y1 = max(0, min(Hh, y1 + pad))
    if y1 <= y0:
        return 0.0
    band = np.zeros_like(person_mask, dtype=bool)
    band[y0:y1, :] = True
    cols = np.any(person_mask, axis=0)
    cols2 = np.repeat(cols[np.newaxis, :], Hh, axis=0)
    target = band & cols2
    if target.sum() == 0:
        return 0.0
    return float((support_mask & target).sum()) / float(target.sum())

# ---------------- Temporal smoothing ----------------
def smooth_posture(state: SceneGraphState, person_id: str, raw_posture: str, window: int):
    if raw_posture is None:
        raw_posture = "unknown"
    hist = state.posture_history.get(person_id)
    if hist is None:
        hist = deque(maxlen=window)
        state.posture_history[person_id] = hist
    hist.append(raw_posture)

    if all(p == "unknown" for p in hist):
        return "unknown"

    counts = defaultdict(int)
    for p in hist:
        if p != "unknown":
            counts[p] += 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "unknown"

def smooth_support(state: SceneGraphState, person_id: str, raw_support: Optional[str],
                   window: int, confirm: int):
    if raw_support is None:
        raw_support = "none"
    hist = state.support_history.get(person_id)
    if hist is None:
        hist = deque(maxlen=window)
        state.support_history[person_id] = hist
    hist.append(raw_support)

    counts = defaultdict(int)
    for s in hist:
        counts[s] += 1
    best_support, freq = max(counts.items(), key=lambda x: x[1])

    if freq < confirm:
        return hist[-2] if len(hist) > 1 else raw_support
    return best_support

def smooth_relation(state: SceneGraphState, subj: str, obj: str, rel: str,
                    window: int, confirm: int):
    key = (subj, obj)
    hist = state.rel_history.get(key)
    if hist is None:
        hist = deque(maxlen=window)
        state.rel_history[key] = hist
    hist.append(rel)

    counts = defaultdict(int)
    for r in hist:
        counts[r] += 1
    best_rel, freq = max(counts.items(), key=lambda x: x[1])
    if freq < confirm:
        return hist[-2] if len(hist) > 1 else rel
    return best_rel

# ---------------- Relations ----------------
def pair_proximity(a: AlignedDet, b: AlignedDet, z, img_wh,
                   prox_px_alpha: float, prox_dz_alpha: float, depth_rng: float):
    Ww, Hh = img_wh
    ax, ay = centroid(a.mask)
    bx, by = centroid(b.mask)
    if any(np.isnan([ax, ay, bx, by])):
        return 0.0

    dist = np.hypot((ax - bx) / max(1e-6, Ww), (ay - by) / max(1e-6, Hh))
    prox_px = math.exp(-dist / prox_px_alpha)

    if np.isfinite(a.depth_median) and np.isfinite(b.depth_median):
        dz = abs(a.depth_median - b.depth_median) / max(1e-6, depth_rng)
        prox_dz = math.exp(-dz / prox_dz_alpha)
    else:
        prox_dz = 0.5

    return 0.5 * prox_px + 0.5 * prox_dz

def left_right_above_below(a: AlignedDet, b: AlignedDet, z, img_wh,
                           prox_min: float,
                           lr_depth_overlap: float,
                           lr_horiz_eps: float,
                           above_vert_eps: float,
                           prox_px_alpha: float,
                           prox_dz_alpha: float,
                           depth_rng: float):
    Ww, Hh = img_wh
    if pair_proximity(a, b, z, (Ww, Hh), prox_px_alpha, prox_dz_alpha, depth_rng) < prox_min:
        return []

    aM = dilate_bool(a.mask, 3)
    bM = dilate_bool(b.mask, 3)
    ax, ay = centroid(aM)
    bx, by = centroid(bM)
    if any(np.isnan([ax, ay, bx, by])):
        return []

    rels = []
    depth_ok = overlap_in_depth(aM, bM, z, lr_depth_overlap)
    if not depth_ok:
        za = a.depth_median
        zb = b.depth_median
        if np.isfinite(za) and np.isfinite(zb):
            depth_ok = (abs(za - zb) <= 0.01 * depth_rng)

    dx = (bx - ax) / max(1e-6, Ww)
    if depth_ok:
        if dx > lr_horiz_eps:
            rels.append("to the right of")
        elif dx < -lr_horiz_eps:
            rels.append("to the left of")

    if abs(bx - ax) / max(1e-6, Ww) < 0.12 and depth_ok:
        dy = (by - ay) / max(1e-6, Hh)
        if dy > above_vert_eps:
            rels.append("above")
        elif dy < -above_vert_eps:
            rels.append("below")

    return rels

def front_behind_adaptive(a: AlignedDet, b: AlignedDet, z, img_wh,
                          prox_min: float,
                          fb_iou_thr: float,
                          fb_dz_thr: float,
                          x_tol: float,
                          prox_px_alpha: float,
                          prox_dz_alpha: float,
                          depth_rng: float,
                          closer_is_smaller: bool):
    Ww, Hh = img_wh
    if pair_proximity(a, b, z, (Ww, Hh), prox_px_alpha, prox_dz_alpha, depth_rng) < prox_min:
        return None

    ax, ay = centroid(a.mask)
    bx, by = centroid(b.mask)
    if np.isfinite(ax) and np.isfinite(bx):
        dx = abs(ax - bx) / max(1e-6, Ww)
        if dx > x_tol:
            return None

    aM = dilate_bool(a.mask, 3)
    bM = dilate_bool(b.mask, 3)
    inter = (aM & bM)
    denom = (aM | bM).sum()

    if denom > 0 and inter.sum() / denom >= fb_iou_thr and inter.any():
        za = float(np.median(z[inter & aM])) if (inter & aM).any() else np.nan
        zb = float(np.median(z[inter & bM])) if (inter & bM).any() else np.nan
        if np.isfinite(za) and np.isfinite(zb):
            dz = (zb - za) if closer_is_smaller else (za - zb)
            if abs(dz) / max(1e-6, depth_rng) > fb_dz_thr:
                return (b, "behind", a) if dz > 0 else (a, "behind", b)

    ma = float(np.median(z[a.mask])) if a.mask.any() else np.nan
    mb = float(np.median(z[b.mask])) if b.mask.any() else np.nan
    if not (np.isfinite(ma) and np.isfinite(mb)):
        return None

    dz = (mb - ma) if closer_is_smaller else (ma - mb)
    if abs(dz) / max(1e-6, depth_rng) > 0.28:
        return (b, "behind", a) if dz > 0 else (a, "behind", b)

    return None

def swap_lr(rel: str, flip_left_right: bool) -> str:
    if not flip_left_right:
        return rel
    if rel == "to the left of":
        return "to the right of"
    if rel == "to the right of":
        return "to the left of"
    return rel

def swap_fb(rel: str, flip_front_back: bool) -> str:
    if not flip_front_back:
        return rel
    if rel == "behind":
        return "in front of"
    return rel

# ---------------- Mask alignment + floor smoothing ----------------
def align_masks_to_detections(
    boxes, labels, masks_pack: MaskPack, img_wh, depth_map,
    img_gray: np.ndarray,
    state: SceneGraphState,
    depth_rng: float,
    floor_ema_alpha: float,
    closer_is_smaller: bool,
):
    Ww, Hh = img_wh
    aligned: List[AlignedDet] = []
    used = np.zeros(len(masks_pack.instances), dtype=bool)

    # Floor params with EMA
    if masks_pack.semantic is not None and depth_map is not None:
        floor_mask_global = carve_semantic_in_box(masks_pack.semantic, (0, 0, Ww, Hh), set(["floor","carpet","rug","ground","mat"]))
        state.floor_mask_global = floor_mask_global

        if floor_mask_global.sum() > 0:
            raw_floor = ransac_floor_plane(depth_map, floor_mask_global, (Ww, Hh), depth_rng)
            if raw_floor != (0.0, 0.0, 0.0):
                if state.prev_floor_params is None or state.prev_floor_params == (0.0, 0.0, 0.0):
                    state.floor_params = raw_floor
                else:
                    state.floor_params = tuple(
                        float(floor_ema_alpha * pf + (1.0 - floor_ema_alpha) * rf)
                        for pf, rf in zip(state.prev_floor_params, raw_floor)
                    )
                state.prev_floor_params = state.floor_params
            else:
                state.floor_params = state.prev_floor_params or (0.0, 0.0, 0.0)
        else:
            state.floor_params = state.prev_floor_params or (0.0, 0.0, 0.0)
    else:
        state.floor_mask_global = np.zeros((Hh, Ww), dtype=bool)
        state.floor_params = state.prev_floor_params or (0.0, 0.0, 0.0)

    for box, label in zip(boxes, labels):
        best_j, best_s, best_iou = -1, -1e9, 0.0
        for j, inst in enumerate(masks_pack.instances):
            if used[j]:
                continue
            iou_val = mask_bbox_iou(inst.mask, box)
            s = iou_val
            if label_compatible(label, inst.label):
                s += 0.2
            s += 0.05 * inst.score
            if s > best_s:
                best_j, best_s, best_iou = j, s, iou_val

        if (best_j >= 0 and best_s > 0.20 and best_iou > 0.15):
            m = masks_pack.instances[best_j].mask.copy()
            used[best_j] = True
            src = "mask2former"
        else:
            m = bbox_to_mask(box, (Hh, Ww))
            src = "bbox_fallback"

        # semantic carve floor/nightstand
        if masks_pack.semantic is not None:
            sem = masks_pack.semantic
            if is_floor(label) and state.floor_params != (0.0, 0.0, 0.0):
                m = state.floor_mask_global & bbox_to_mask(box, (Hh, Ww))
                src = "semantic_floor_ransac"
            elif is_nightstand(label):
                mm = carve_semantic_in_box(sem, box, set(["nightstand","cabinet","drawer","dresser","chest","side table","end table"]))
                if mm.sum() > 0:
                    mm = rectify_mask_in_box_with_edges(mm, box, img_gray)
                    m, src = mm, "semantic_nightstand"

        dp_med = float(np.median(depth_map[m])) if (depth_map is not None and m.sum() > 0) else float("nan")

        aligned.append(
            AlignedDet(
                label=label,
                score=0.0,
                box=tuple(map(float, box)),
                mask=m,
                source=src,
                depth_median=dp_med,
            )
        )

    return aligned

# ---------------- Main scene graph computation ----------------
def compute_scene_graph_with_masks(
    aligned: List[AlignedDet],
    z: np.ndarray,
    img_wh: Tuple[int, int],
    poses: List[Dict[str, Any]],
    state: SceneGraphState,
    cfg: Dict[str, Any],
    depth_rng: float,
):
    Ww, Hh = img_wh
    scene: List[Tuple[str, str, str]] = []

    prox_min = cfg["proximity"]["prox_min"]
    prox_px_alpha = cfg["proximity"]["prox_px_alpha"]
    prox_dz_alpha = cfg["proximity"]["prox_dz_alpha"]

    lr_depth_overlap = cfg["spatial"]["lr_depth_overlap"]
    lr_horiz_eps = cfg["spatial"]["lr_horiz_eps"]
    above_vert_eps = cfg["spatial"]["above_vert_eps"]

    fb_iou_thr = cfg["spatial"]["fb_iou_thr"]
    fb_dz_thr = cfg["spatial"]["fb_dz_thr"]
    closer_is_smaller = cfg["spatial"]["closer_is_smaller"]

    flip_lr = cfg["relations"]["flip_left_right"]
    flip_fb = cfg["relations"]["flip_front_back"]

    rel_window = cfg["smoothing"]["rel_smooth_window"]
    rel_confirm = cfg["smoothing"]["rel_confirm"]

    # pairwise spatial relations (ignore floor)
    for i in range(len(aligned)):
        a = aligned[i]
        for j in range(i + 1, len(aligned)):
            b = aligned[j]
            if is_floor(a.label) or is_floor(b.label):
                continue

            rels = left_right_above_below(
                a, b, z, (Ww, Hh),
                prox_min, lr_depth_overlap, lr_horiz_eps, above_vert_eps,
                prox_px_alpha, prox_dz_alpha, depth_rng
            )
            for r in rels:
                rr = swap_lr(r, flip_lr)
                rr = smooth_relation(state, a.label, b.label, rr, rel_window, rel_confirm)
                scene.append((a.label, rr, b.label))

            fb = front_behind_adaptive(
                a, b, z, (Ww, Hh),
                prox_min, fb_iou_thr, fb_dz_thr, x_tol=0.20,
                prox_px_alpha=prox_px_alpha, prox_dz_alpha=prox_dz_alpha,
                depth_rng=depth_rng, closer_is_smaller=closer_is_smaller
            )
            if fb is not None:
                subj, rel, obj = fb
                rr = swap_fb(rel, flip_fb)
                rr = smooth_relation(state, subj.label, obj.label, rr, rel_window, rel_confirm)
                scene.append((subj.label, rr, obj.label))

    def match_pose_to_person(p_box):
        best, best_s = None, -1e9
        for po in poses:
            b = po["bbox"]
            i = iou(p_box, b)
            px = (p_box[0] + p_box[2]) / 2
            py = (p_box[1] + p_box[3]) / 2
            bx = (b[0] + b[2]) / 2
            by = (b[1] + b[3]) / 2
            cd = np.hypot(px - bx, py - by) / (np.hypot(p_box[2] - p_box[0], p_box[3] - p_box[1]) + 1e-6)
            s = 0.7 * i + 0.3 * (1.0 - min(1.0, cd))
            if s > best_s:
                best, best_s = po, s
        return best

    posture_window = cfg["smoothing"]["posture_smooth_window"]
    support_window = cfg["smoothing"]["support_smooth_window"]
    support_confirm = cfg["smoothing"]["support_change_confirm"]

    ransac_floor_thr = cfg["ransac"]["ransac_floor_thr"]

    people = [d for d in aligned if "person" in d.label.lower()]
    supports_idx = [
        k for k, d in enumerate(aligned)
        if any(t in d.label.lower() for t in [
            "chair","armchair","stool","bench","sofa","couch","bed","floor","carpet","rug","mat","ground"
        ])
    ]

    for det in people:
        po = match_pose_to_person(det.box)
        if po is None:
            continue
        kpts = np.asarray(po["kpts"])
        raw_posture = classify_posture_from_kpts(kpts)
        posture = smooth_posture(state, det.label, raw_posture, posture_window)
        if posture == "unknown":
            continue

        px0, py0, px1, py1 = map(int, det.box)
        ph = max(1, py1 - py0)
        if posture == "standing_on":
            cy = py1 - max(2, int(0.02 * ph))
        elif posture == "sitting_on":
            cy = py0 + int(0.55 * ph)
        else:
            cy = py0 + ph // 2
        y0, y1 = cy - max(2, int(0.02 * ph)) // 2, cy + max(2, int(0.02 * ph)) // 2

        # floor confirmation via RANSAC plane
        best_floor = None
        if state.floor_params != (0.0, 0.0, 0.0) and posture in ("lying_on","standing_on","falling"):
            cx, cy_p = centroid(det.mask)
            if np.isfinite(cx) and np.isfinite(det.depth_median):
                dz_from_plane = depth_diff_to_plane(
                    cx, cy_p, det.depth_median, state.floor_params, img_wh,
                    closer_is_smaller=closer_is_smaller
                )
                if dz_from_plane >= -ransac_floor_thr and dz_from_plane < 2.5 * ransac_floor_thr:
                    for idx in supports_idx:
                        sdet = aligned[idx]
                        if is_floor(sdet.label):
                            best_floor = (10.0, sdet)
                            break

        # fallback floor overlap/underlap
        if best_floor is None:
            floor_scores = []
            for idx in supports_idx:
                sdet = aligned[idx]
                if not is_floor(sdet.label):
                    continue
                ov = (det.mask & sdet.mask).sum() / (det.mask.sum() + 1e-6)
                ul = underlap_ratio(det.mask, sdet.mask, y0, y1, pad=8)
                floor_scores.append((0.9 * ov + 1.1 * ul, sdet))
            if floor_scores:
                best_floor = max(floor_scores, key=lambda x: x[0])

        # falling_towards floor
        if posture == "falling" and best_floor is not None and best_floor[0] >= 0.10:
            scene.append((det.label, "falling_towards", best_floor[1].label))
            continue

        # default: posture support assignment
        if posture in ("lying_on","standing_on") and best_floor is not None and best_floor[0] >= 0.10:
            scene.append((det.label, posture, best_floor[1].label))
            continue

        # non-floor support candidates
        candidates = []
        for idx in supports_idx:
            sdet = aligned[idx]
            L = sdet.label.lower()
            inter = (det.mask & sdet.mask).sum() / (det.mask.sum() + 1e-6)
            horiz_cols = (np.any(det.mask, axis=0) & np.any(sdet.mask, axis=0)).sum() / (np.any(det.mask, axis=0).sum() + 1e-6)

            ul = underlap_ratio(det.mask, sdet.mask, y0, y1, pad=10 if posture == "lying_on" else 6)
            c_ratio = float((det.mask & sdet.mask).sum()) / (det.mask.sum() + 1e-6)
            iou_pb = iou(det.box, sdet.box)
            mask_quality = 1.0 if sdet.source != "bbox_fallback" else 0.35

            score_sup = (1.25 * ul + 0.55 * c_ratio + 0.20 * iou_pb + 0.25 * horiz_cols) * mask_quality

            if posture == "standing_on":
                score_sup += (0.35 if is_floor(sdet.label) else -0.25)
            elif posture == "lying_on":
                if any(k in L for k in ["bed","sofa","couch"]):
                    score_sup += 0.10
                if is_floor(sdet.label):
                    score_sup += 0.25

            candidates.append((score_sup, sdet))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_sup = candidates[0][1]
        final_support = smooth_support(state, det.label, best_sup.label, support_window, support_confirm)
        scene.append((det.label, posture, final_support))

    # dedup
    out = []
    seen = set()
    for t in scene:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

# ---------------- Main public API: per-frame scene graph ----------------
def run_scene_graph_for_frame(
    frame_bgr: np.ndarray,
    models,
    state: SceneGraphState,
    cfg: Dict[str, Any],
):
    device = models.device
    W = frame_bgr.shape[1]
    H = frame_bgr.shape[0]

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)
    img_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # tracking for stable person ids
    track_boxes, track_ids = get_tracked(frame_bgr, models.tracker_model, device)

    # DINO
    caption = cfg["caption"]["text"]
    thr = cfg["dino_thresholds"]
    inputs = models.dino_processor(images=image, text=caption, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = models.dino_model(**inputs)

    results = models.dino_processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        threshold=thr["logit_thr"],
        text_threshold=thr["text_thr"],
        target_sizes=[(H, W)],
    )[0]

    boxes_np  = results["boxes"].detach().cpu().numpy()
    scores_np = results["scores"].detach().cpu().numpy()
    labels_raw = results.get("text_labels", results.get("labels", []))

    keep = [i for i, s in enumerate(scores_np) if float(s) > thr["box_conf_thr"]]
    boxes = [boxes_np[i].tolist() for i in keep]
    raw_labels = [labels_raw[i] for i in keep]

    # permanent label assignment
    final_labels = []
    used_track_ids = set()

    for box, raw_label in zip(boxes, raw_labels):
        best_i = 0.0
        best_tid = None
        for tbox, tid in zip(track_boxes, track_ids):
            if tid in used_track_ids:
                continue
            ii = iou(box, tbox.tolist())
            if ii > best_i and ii > 0.5:
                best_i = ii
                best_tid = int(tid)

        label_lower = str(raw_label).lower().split()[0]

        if any(w in label_lower for w in ["floor","carpet","rug","mat","ground"]):
            if state.eternal_floor_id is None:
                state.eternal_floor_id = "floor_0"
            final_label = state.eternal_floor_id
        elif "bed" in label_lower:
            if state.eternal_bed_id is None:
                state.eternal_bed_id = "bed_0"
            final_label = state.eternal_bed_id
        elif any(w in label_lower for w in ["sofa","couch"]):
            if state.eternal_sofa_id is None:
                state.eternal_sofa_id = "sofa_0"
            final_label = state.eternal_sofa_id
        elif "person" in label_lower:
            if best_tid is not None:
                if best_tid not in state.person_id_map:
                    state.person_id_map[best_tid] = f"person_{state.next_person_id}"
                    state.next_person_id += 1
                final_label = state.person_id_map[best_tid]
            else:
                final_label = f"person_{state.next_person_id}"
                state.next_person_id += 1
        else:
            base = label_lower
            count = sum(1 for l in final_labels if l.startswith(base))
            final_label = f"{base}_{count}"

        final_labels.append(final_label)
        if best_tid is not None:
            used_track_ids.add(best_tid)

    # depth
    dcfg = cfg["ransac"]
    with torch.no_grad():
        dpt_in = models.dpt_fe(images=image, return_tensors="pt").to(device)
        depth  = models.dpt_model(**dpt_in).predicted_depth

    depth_raw = (
        torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if state.prev_depth_map is None or state.prev_depth_map.shape != depth_raw.shape:
        depth_map = depth_raw
    else:
        a = float(dcfg["depth_ema_alpha"])
        depth_map = (a * state.prev_depth_map + (1.0 - a) * depth_raw).astype(np.float32)

    state.prev_depth_map = depth_map.copy()
    depth_rng = max(1e-6, float(depth_map.max() - depth_map.min()))

    # pose + masks
    poses = yolo_pose_people(frame_bgr, models.pose_model)
    masks_pack = build_instance_and_semantic_masks(
        image_pil=image,
        device=device,
        mask2former_proc=models.mask2former_proc,
        mask2former_model=models.mask2former_model,
        segformer_proc=models.segformer_proc,
        segformer_model=models.segformer_model,
        img_gray=img_gray
    )

    aligned = align_masks_to_detections(
        boxes, final_labels, masks_pack, (W, H), depth_map,
        img_gray=img_gray,
        state=state,
        depth_rng=depth_rng,
        floor_ema_alpha=float(dcfg["floor_ema_alpha"]),
        closer_is_smaller=cfg["spatial"]["closer_is_smaller"],
    )

    triples = compute_scene_graph_with_masks(
        aligned, depth_map, (W, H), poses,
        state=state,
        cfg=cfg.raw if hasattr(cfg, "raw") else cfg,
        depth_rng=depth_rng
    )

    scene_graph_json = [{"subject": s, "relation": r, "object": o} for (s, r, o) in triples]
    return scene_graph_json, frame_bgr, depth_rng
