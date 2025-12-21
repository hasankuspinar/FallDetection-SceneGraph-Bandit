import numpy as np

def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    bb = max(0, bx1 - bx0) * max(0, by1 - by0)
    return inter / max(1e-6, aa + bb - inter)

def bbox_to_mask(box, shape):
    x0, y0, x1, y1 = map(int, box)
    h, w = shape
    x0, x1 = np.clip([x0, x1], 0, w)
    y0, y1 = np.clip([y0, y1], 0, h)
    m = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m

def centroid(m):
    ys, xs = np.where(m)
    return (float(xs.mean()), float(ys.mean())) if xs.size else (np.nan, np.nan)
