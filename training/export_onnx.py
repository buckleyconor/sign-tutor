import torch
import torch.nn as nn
from training.model_one_hand import OneHandClassifier
from training.model_two_hand import TwoHandClassifier


def export_one_hand(model_path: str, output_path: str, num_classes: int = 26):
    model = OneHandClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    dummy = torch.randn(1, 63)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported one-hand model to {output_path}")


def export_two_hand(model_path: str, output_path: str, num_classes: int = 26):
    model = TwoHandClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    dummy = torch.randn(1, 126)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported two-hand model to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="model.onnx")
    parser.add_argument("--hands", type=int, choices=[1, 2], default=1)
    parser.add_argument("--num-classes", type=int, default=26)
    args = parser.parse_args()
    if args.hands == 1:
        export_one_hand(args.checkpoint, args.output, args.num_classes)
    else:
        export_two_hand(args.checkpoint, args.output, args.num_classes)
