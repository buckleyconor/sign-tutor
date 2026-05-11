import pandas as pd
import sys
import numpy as np


def check(path: str, expected_dim: int) -> int:
    df = pd.read_csv(path, header=None)
    n_meta = 3  # label, source, frame_id
    feature_cols = df.shape[1] - n_meta
    failures = []

    if feature_cols != expected_dim:
        failures.append(f"Wrong dim: got {feature_cols}, expected {expected_dim}")

    counts = df.iloc[:, 0].value_counts()
    ratio = counts.min() / counts.max()
    if ratio < 0.5:
        failures.append(f"Imbalance: min/max class ratio = {ratio:.2f}")

    feats = df.iloc[:, n_meta:].astype(float)
    if feats.isna().any().any():
        failures.append("NaN values present")
    if np.isinf(feats.to_numpy()).any():
        failures.append("Inf values present")

    sources = df.iloc[:, 1].nunique()
    if sources < 2:
        failures.append(f"Only {sources} source(s); want >= 2")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("Data quality: OK")
    return 0


if __name__ == "__main__":
    sys.exit(check(sys.argv[1], int(sys.argv[2])))
