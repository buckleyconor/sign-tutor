import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, max_hands: int = 2,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5,
                 static_image_mode: bool = False):
        """Hand tracker wrapper around MediaPipe Hands.

        Args:
            max_hands: Maximum number of hands to detect (default 2).
            min_detection_confidence: Confidence threshold for initial detection.
            min_tracking_confidence: Confidence threshold for continued tracking.
            static_image_mode: If True, re-run detection on every frame (no temporal
                tracking). Use for static image batch processing; leave False for
                live video/webcam feeds.
        """
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray):
        """Returns list of (handedness, landmarks_array) tuples.
        landmarks_array is shape (21, 3)."""
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        out = []
        for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[hand_idx].classification[0].label
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            out.append((handedness, arr))
        return out

    def close(self):
        self._hands.close()

    def __del__(self):
        self.close()
