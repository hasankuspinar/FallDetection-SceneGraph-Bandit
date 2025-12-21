import numpy as np

def get_tracked(frame_bgr: np.ndarray, tracker_model, device: str):
    """
    Runs YOLOv8 tracking (BoT-SORT) and returns (boxes, ids) for persons.
    boxes: (N,4) xyxy, ids: (N,) int
    """
    res = tracker_model.track(
        frame_bgr,
        persist=True,
        tracker="botsort.yaml",
        classes=[0],      # 0: person
        conf=0.3,
        iou=0.5,
        verbose=False,
        device=device,
    )[0]

    if len(res.boxes):
        boxes = res.boxes.xyxy.cpu().numpy()
    else:
        boxes = np.empty((0, 4))

    if res.boxes.id is not None:
        ids = res.boxes.id.int().cpu().numpy()
    else:
        ids = np.array([], dtype=int)

    return boxes, ids
