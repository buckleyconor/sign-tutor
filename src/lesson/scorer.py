import time

_EMA_ALPHA = 0.25  # per-frame smoothing — lower = smoother but slower to respond


class QualityScorer:
    """Confidence quality bar: EMA-smoothed, red→green, full green at green_min."""

    def __init__(self, target_idx: int, hold_seconds: float = 1.0,
                 green_min: float = 0.90):
        self._target = target_idx
        self._hold = hold_seconds
        self._green_min = green_min
        self._green_since: float | None = None
        self._ema: float = 0.0

    def update(self, pred_idx: int, conf: float):
        """Feed a new prediction; updates EMA and completion timer."""
        target_conf = conf if pred_idx == self._target else 0.0
        self._ema += _EMA_ALPHA * (target_conf - self._ema)
        now = time.monotonic()
        if pred_idx == self._target and self._ema >= self._green_min:
            if self._green_since is None:
                self._green_since = now
        else:
            self._green_since = None

    @property
    def quality(self) -> float:
        """Smoothed quality value 0.0–1.0 for display."""
        return self._ema

    @property
    def completed(self) -> bool:
        return (
            self._green_since is not None
            and time.monotonic() - self._green_since >= self._hold
        )

    def render_bar(self) -> str:
        """Return HTML quality bar. Reaches full green at green_min confidence."""
        pct = self._ema                          # 0.0–1.0 smoothed confidence
        display = min(pct / self._green_min, 1.0)  # 0–1, saturates at threshold
        hue = int(display * 120)                 # 0=red, 60=yellow, 120=green
        bar_w = int(display * 100)
        label = int(pct * 100)
        color = f"hsl({hue},80%,42%)"
        return (
            '<div style="background:#e0e0e0;border-radius:8px;height:36px;'
            'position:relative;overflow:hidden;margin:4px 0;">'
            f'<div style="width:{bar_w}%;height:100%;background:{color};'
            'transition:width 0.12s ease,background 0.12s ease;"></div>'
            '<span style="position:absolute;inset:0;display:flex;align-items:center;'
            f'justify-content:center;font-weight:bold;font-size:15px;color:#111;">'
            f'{label}%</span>'
            '</div>'
        )
