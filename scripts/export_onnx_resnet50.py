#!/usr/bin/env python3
import torch, torchvision, argparse, onnx
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--batch", type=int, default=32)
args = parser.parse_args()

model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
model.eval()
dummy = torch.randn(args.batch, 3, 224, 224)
torch.onnx.export(model, dummy, args.out, input_names=["input"], output_names=["output"],
                  opset_version=13, dynamic_axes={"input": {0:"batch"}, "output": {0:"batch"}})
print(f"Saved ONNX to {args.out}")
