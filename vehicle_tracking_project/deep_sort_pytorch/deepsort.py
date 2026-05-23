
import numpy as np
from .deep.feature_extractor import Extractor
from .deep_sort.detection import Detection
from .deep_sort.nn_matching import NearestNeighborDistanceMetric
from .deep_sort.tracker import Tracker

__all__ = ["DeepSort"]

class DeepSort:
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3,
                 max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.extractor      = Extractor(model_path, use_cuda=use_cuda)
        metric              = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker        = Tracker(metric, max_iou_distance=max_iou_distance,
                                      max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        h, w = ori_img.shape[:2]
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i])
            for i, conf in enumerate(confidences)
            if conf > self.min_confidence
        ]
        self.tracker.predict()
        self.tracker.update(detections)
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlbr()
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            track_id = track.track_id
            outputs.append([x1, y1, x2, y2, track_id])
        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        bbox_tlwh = bbox_xywh.copy().astype(float)
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _get_features(self, bbox_xywh, ori_img):
        h, w = ori_img.shape[:2]
        im_crops = []
        for box in bbox_xywh:
            x1 = max(0, int(box[0] - box[2]/2))
            y1 = max(0, int(box[1] - box[3]/2))
            x2 = min(w, int(box[0] + box[2]/2))
            y2 = min(h, int(box[1] + box[3]/2))
            im_crops.append(ori_img[y1:y2, x1:x2])
        return self.extractor(im_crops) if im_crops else []
