from collections import deque, Counter
import numpy as np


class PredictionSmoother:
    def __init__(self, window: int = 15):
        self._window = window
        self._preds = deque(maxlen=window)
        self._confs = deque(maxlen=window)

    def update(self, pred_idx: int, confidence: float):
        self._preds.append(pred_idx)
        self._confs.append(confidence)

    def smoothed(self) -> tuple[int, float] | None:
        if len(self._preds) < self._window // 2:
            return None
        modal, count = Counter(self._preds).most_common(1)[0]
        avg_conf = float(np.mean(
            [c for p, c in zip(self._preds, self._confs) if p == modal]
        ))
        return modal, avg_conf
