import os, json, time
from typing import Dict, List, Optional, Tuple
import cv2

from .config import cfg_get, Config
from .models import load_models
from .scene_graph import SceneGraphState, run_scene_graph_for_frame
from .bandit import (
    delta_score, derive_reason, translate_delta,
    ConstraintManager, ConstrainedLinUCB,
    feature_vector_from_deltas, TwoFrameState, KOvN
)
import math as _math

def significant_change_between_graphs(
    prev_graph: List[Dict],
    cur_graph: List[Dict],
    state: TwoFrameState,
    bandit_policy: ConstrainedLinUCB,
    constraint_mgr: ConstraintManager,
    weights: Dict[str, float],
    score_thr: float,
    cooldown_s: float,
    summary_max: int,
) -> Dict:
    score, notes = delta_score(prev_graph, cur_graph, weights=weights)
    k_fire = state.smoother.push(score)

    delta_text = translate_delta(1, notes, cur_graph, max_lines=summary_max)
    reason_hint = derive_reason(notes, score, thr=score_thr)

    x = feature_vector_from_deltas(score, notes, cur_graph)
    action, conf, bandit_reason = bandit_policy.decide(
        x, constraint_mgr, cooldown_ok=state.cooldown_ok(cooldown_s)
    )

    result = {
        "significant_change": bool(action == 1 and k_fire),
        "confidence": float(conf),
        "reason": f"{bandit_reason}: {reason_hint}"[:300],
        "score": float(score),
        "delta_text": delta_text,
        "dynamic_threshold": float(state.dynamic_thr.value),
    }

    if result["significant_change"]:
        constraint_mgr.on_fire()
        state.mark_fired()

    thr = state.dynamic_thr.value
    if score >= thr and result["significant_change"]:
        rew, outcome = 1.0, "TP"
    elif score < thr and result["significant_change"]:
        rew, outcome = -0.3, "FP"
    elif score >= thr and not result["significant_change"]:
        rew, outcome = -1.0, "FN"
    else:
        rew, outcome = 0.1, "TN"

    bandit_policy.update(x, 1 if result["significant_change"] else 0, rew)
    state.dynamic_thr.update(outcome)

    return result

def compute_fall_score(results: List[dict]) -> Dict[str, float]:
    if not results:
        return {"fall_score": 0.0, "posture_ratio": 0.0, "floor_ratio": 0.0,
                "support_ratio": 0.0, "duration_ratio": 0.0, "depth_impact": 0.0}

    posture_drop = 0
    floor_contact = 0
    support_flip = 0
    on_floor_frames = 0
    total = len(results)

    for r in results:
        reason = r.get("reason", "").lower()
        delta = r.get("delta_text", "").lower()

        if ("posture drop" in reason) or ("standing/sitting→lying" in delta) or ("posture=lying_on" in delta):
            posture_drop += 1
        if "floor contact" in reason or "support → floor" in delta:
            floor_contact += 1
        if "support flip" in reason and "floor" in reason:
            support_flip += 1
        if ("on_floor=true" in reason) or ("posture=lying_on" in delta) or ("lying_on" in reason):
            on_floor_frames += 1
        if "falling_towards floor" in reason or "falling_towards" in delta:
            posture_drop += 1

    posture_ratio = posture_drop / max(1, total)
    floor_ratio   = floor_contact / max(1, total)
    support_ratio = support_flip / max(1, total)
    duration_ratio = on_floor_frames / max(1, total)

    # weights (your validated defaults)
    w_posture  = 3.5
    w_floor    = 1.5
    w_support  = 0.0
    w_duration = 1.5
    baseline_offset = -1.0

    duration_boost_thr = 0.02
    duration_boost_val = 0.2
    duration_boost = duration_boost_val if duration_ratio > duration_boost_thr else 0.0

    raw_score = (
        w_posture  * posture_ratio +
        w_floor    * floor_ratio +
        w_support  * support_ratio +
        w_duration * (duration_ratio + duration_boost) +
        baseline_offset
    )
    fall_score = 1.0 / (1.0 + _math.exp(-raw_score))

    return {
        "posture_ratio": round(posture_ratio, 3),
        "floor_ratio": round(floor_ratio, 3),
        "support_ratio": round(support_ratio, 3),
        "duration_ratio": round(duration_ratio, 3),
        "depth_impact": 0.0,
        "fall_score": round(fall_score, 3),
    }

def process_video(video_path: str, fps_target: int, logs_dir: str, cfg: Config):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    # config values
    prefer_cuda = bool(cfg_get(cfg, "device", "prefer_cuda", default=True))
    dino_model_id = cfg_get(cfg, "models", "dino_model_id")
    dpt_model_id = cfg_get(cfg, "models", "dpt_model_id")
    mask2former_id = cfg_get(cfg, "models", "mask2former_id")
    segformer_ade_id = cfg_get(cfg, "models", "segformer_ade_id")
    yolo_pose_ckpt = cfg_get(cfg, "models", "yolo_pose_ckpt", default="yolov8s-pose.pt")
    yolo_track_ckpt = cfg_get(cfg, "models", "yolo_track_ckpt", default="yolov8x.pt")

    models = load_models(
        dino_model_id=dino_model_id,
        dpt_model_id=dpt_model_id,
        mask2former_id=mask2former_id,
        segformer_ade_id=segformer_ade_id,
        yolo_pose_ckpt=yolo_pose_ckpt,
        yolo_track_ckpt=yolo_track_ckpt,
        prefer_cuda=prefer_cuda,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(max(fps, 1e-3) / max(fps_target, 1))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[video] {video_path} total_frames={total_frames} fps={fps:.2f} sampling every {step} frames (≈{fps_target}fps) 🎬")

    sg_state = SceneGraphState()

    # bandit setup
    score_thr = float(cfg_get(cfg, "bandit", "score_thr", default=2.0))
    cooldown_s = float(cfg_get(cfg, "bandit", "cooldown_s", default=4.0))
    summary_max = int(cfg_get(cfg, "bandit", "summary_max", default=6))

    k = int(cfg_get(cfg, "bandit", "k_of_n", "k", default=1))
    n = int(cfg_get(cfg, "bandit", "k_of_n", "n", default=1))

    weights = {
        "posture_drop": 3.0,
        "floor_contact": 3.0,
        "support_flip": 1.4,
        "person_birth": 0.6,
        "person_death": 0.4,
        "lr_above_below": 0.4,
        "falling": 2.5,
    }

    alpha = float(cfg_get(cfg, "bandit", "alpha", default=1.4))
    eps = float(cfg_get(cfg, "bandit", "eps_greedy", default=0.05))

    bandit_policy = ConstrainedLinUCB(d=13, alpha=alpha, seed=123, eps_greedy=eps)

    cm = ConstraintManager(
        max_alarms_per_minute=int(cfg_get(cfg, "bandit", "constraints", "max_alarms_per_minute", default=6)),
        budget_per_minute=float(cfg_get(cfg, "bandit", "constraints", "budget_per_minute", default=6.0)),
    )

    state = TwoFrameState(smoother=KOvN(k=k, n=n, thr=score_thr))

    prev_graph = None
    results = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        cur_graph_json, _, _ = run_scene_graph_for_frame(frame_bgr, models, sg_state, cfg.raw)

        if prev_graph is not None:
            verdict = significant_change_between_graphs(
                prev_graph, cur_graph_json,
                state=state,
                bandit_policy=bandit_policy,
                constraint_mgr=cm,
                weights=weights,
                score_thr=score_thr,
                cooldown_s=cooldown_s,
                summary_max=summary_max,
            )
            verdict["frame"] = frame_idx
            results.append(verdict)
            print(f"[frame {frame_idx}] verdict: {verdict}")

        prev_graph = cur_graph_json
        frame_idx += 1

    cap.release()

    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, "all_results.json")
    summary = compute_fall_score(results)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"frames": results, "summary": summary}, f, ensure_ascii=False, indent=2)

    print(f"[video] Saved: {out_path} ✅")
    return results, fps, total_frames

def summarize_results(results: list, total_frames: int, fps_target: int):
    print("\nSUMMARY 📌")
    print(f"- Total frames analyzed: {total_frames} (sampled at {fps_target} FPS)")
    significant_changes = [r for r in results if r.get("significant_change", False)]
    if significant_changes:
        print(f"- Significant changes detected: {len(significant_changes)}")
        for change in significant_changes[:10]:
            print(f"  * Frame {change.get('frame')}: {change.get('reason')} (conf={change.get('confidence',0):.2f})")
    else:
        print("- No significant changes detected.")

def summarize_fall_score(results: List[dict], total_frames: int, fps_target: float):
    s = compute_fall_score(results)
    print("\n=== FALL LIKELIHOOD ANALYSIS 🧠 ===")
    print(f"Posture drop ratio:   {s['posture_ratio']:.3f}")
    print(f"Floor contact ratio:  {s['floor_ratio']:.3f}")
    print(f"Support flip ratio:   {s['support_ratio']:.3f}")
    print(f"Duration on floor:    {s['duration_ratio']:.3f}")
    print(f"\n🏁 FALL SCORE (0–1): {s['fall_score']:.3f}")

    if s["fall_score"] > 0.7:
        print("⚠️ Likely fall event detected.")
    elif s["fall_score"] > 0.4:
        print("🟡 Possible fall — verification recommended.")
    else:
        print("🟢 No fall detected.")
