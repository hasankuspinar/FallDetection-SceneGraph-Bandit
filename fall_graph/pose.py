import numpy as np

def _angle(a, b, c):
    ba = a - b
    bc = c - b
    ba /= np.linalg.norm(ba) + 1e-6
    bc /= np.linalg.norm(bc) + 1e-6
    return float(np.degrees(np.arccos(np.clip((ba * bc).sum(), -1, 1))))

def _torso_tilt_deg(k):
    sh  = (k[5, :2] + k[6, :2]) / 2
    hip = (k[11, :2] + k[12, :2]) / 2
    v = sh - hip
    v /= np.linalg.norm(v) + 1e-6
    up = np.array([0, -1.0])
    cos = np.clip((v * up).sum(), -1, 1)
    return float(np.degrees(np.arccos(cos)))

def _knee_bend_deg(k):
    angs = []
    for Hh_, Kk_, Aa_ in [(11, 13, 15), (12, 14, 16)]:
        if np.isfinite(k[[Hh_, Kk_, Aa_], :2]).all():
            angs.append(_angle(k[Hh_, :2], k[Kk_, :2], k[Aa_, :2]))
    return float(np.nanmedian(angs)) if angs else 180.0

def classify_posture_from_kpts(k):
    """
    Extended with intermediate 'falling' state.
    """
    if np.count_nonzero(np.isfinite(k[:, :2])) < 10:
        return "unknown"
    tilt = _torso_tilt_deg(k)
    knee = _knee_bend_deg(k)

    if tilt < 28 and knee > 150:
        return "standing_on"
    if tilt < 48 and 55 <= knee <= 145:
        return "sitting_on"
    if 40 <= tilt <= 80 and knee < 140:
        return "falling"
    if tilt > 55:
        return "lying_on"
    return "unknown"

def yolo_pose_people(img_bgr, pose_model):
    """
    Returns list of {bbox: [x0,y0,x1,y1], kpts: (17,3)} for persons.
    """
    r = pose_model.predict(
        img_bgr[..., ::-1], conf=0.25, iou=0.50, imgsz=960, verbose=False
    )[0]

    if getattr(r, "keypoints", None) is None or len(r.keypoints) == 0:
        return []

    kxy = r.keypoints.xy.detach().cpu().numpy()
    out = []
    for bx, ky in zip(r.boxes.xyxy, kxy):
        k = np.concatenate([ky, np.ones((ky.shape[0], 1), dtype=np.float32)], axis=1)
        out.append({"bbox": bx.detach().cpu().numpy().tolist(), "kpts": k})
    return out
