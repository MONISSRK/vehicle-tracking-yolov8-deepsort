
import numpy as np

class Detection:
    def __init__(self, tlwh, confidence, feature):
        self.tlwh       = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.feature    = np.asarray(feature, dtype=float)

    def to_tlwh(self):
        return self.tlwh.copy()

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
