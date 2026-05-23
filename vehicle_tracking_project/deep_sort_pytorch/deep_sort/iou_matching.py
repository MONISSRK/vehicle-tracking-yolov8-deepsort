
import numpy as np
from . import linear_assignment

def iou(bbox, candidates):
    bbox_tl      = bbox[:2]
    bbox_br      = bbox[:2] + bbox[2:]
    cand_tl      = candidates[:, :2]
    cand_br      = candidates[:, :2] + candidates[:, 2:]
    tl           = np.maximum(bbox_tl, cand_tl)
    br           = np.minimum(bbox_br, cand_br)
    wh           = np.maximum(0., br - tl)
    area_inter   = wh.prod(axis=1)
    area_bbox    = bbox[2:].prod()
    area_cands   = candidates[:, 2:].prod(axis=1)
    return area_inter / (area_bbox + area_cands - area_inter)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, trk_idx in enumerate(track_indices):
        if tracks[trk_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        bbox = tracks[trk_idx].to_tlwh()
        cands = np.asarray([detections[i].to_tlwh() for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, cands)
    return cost_matrix
