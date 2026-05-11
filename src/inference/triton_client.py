import numpy as np
import tritonclient.http as httpclient
from tritonclient.http import InferenceServerClient


class TritonClassifier:
    def __init__(self, url: str = "triton:8000", model_name: str = "asl_classifier"):
        self._client = InferenceServerClient(url=url)
        self._model_name = model_name

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (D,) or (B, D). Returns (num_classes,) softmax-able logits."""
        if x.ndim == 1:
            x = x[None, :]
        inp = httpclient.InferInput("input", x.shape, "FP32")
        inp.set_data_from_numpy(x.astype(np.float32))
        resp = self._client.infer(self._model_name, [inp])
        return resp.as_numpy("output")[0]
