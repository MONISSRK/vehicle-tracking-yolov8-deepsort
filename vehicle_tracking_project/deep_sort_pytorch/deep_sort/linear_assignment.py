
import numpy as np
from scipy.optimize import linear_sum_assignment

INFTY_COST = 1e+5

def min_cost_matching(distance_metric, max_distance, tracks, detections,
                      track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], list(track_indices), list(detection_indices)

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, det_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(det_idx)
    for row, trk_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(trk_idx)
    for row, col in zip(row_indices, col_indices):
        trk_idx = track_indices[row]
        det_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(trk_idx)
            unmatched_detections.append(det_idx)
        else:
            matches.append((trk_idx, det_idx))

    return matches, unmatched_tracks, unmatched_detections

def matching_cascade(distance_metric, max_distance, cascade_depth, tracks,
                     detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []

    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        track_indices_l = [k for k in track_indices
                           if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:
            continue
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections)
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - {k for k, _ in matches})
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(kf, cost_matrix, tracks, detections,
                     track_indices, detection_indices, only_position=False):
    from .kalman_filter import chi2inv95
    gating_dim   = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, trk_idx in enumerate(track_indices):
        track = tracks[trk_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = INFTY_COST
    return cost_matrix
