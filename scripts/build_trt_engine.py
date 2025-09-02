#!/usr/bin/env python3
import argparse, json, os, sys, ctypes, numpy as np
import onnx
import tensorrt as trt

parser = argparse.ArgumentParser()
parser.add_argument("--onnx", required=True)
parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
parser.add_argument("--out", required=True)
parser.add_argument("--config", default="./configs/engine_config.json")
args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

with open(args.config) as f:
    cfg = json.load(f)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser_ = trt.OnnxParser(network, TRT_LOGGER)

with open(args.onnx, "rb") as f:
    if not parser_.parse(f.read()):
        for i in range(parser_.num_errors):
            print(parser_.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(cfg.get("workspace", 2<<30)))

profile = builder.create_optimization_profile()
min_s = tuple(cfg["min_shape"])
opt_s = tuple(cfg["opt_shape"])
max_s = tuple(cfg["max_shape"])
profile.set_shape("input", min_s, opt_s, max_s)
config.add_optimization_profile(profile)

if args.precision == "fp16":
    if not builder.platform_has_fast_fp16:
        print("Warning: platform does not have fast fp16")
    config.set_flag(trt.BuilderFlag.FP16)
elif args.precision == "int8":
    config.set_flag(trt.BuilderFlag.INT8)
    # TODO: attach calibrator implementation (stubbed in C++)
    # config.int8_calibrator = MyCalibrator(...)

engine = builder.build_engine(network, config)
with open(args.out, "wb") as f:
    f.write(engine.serialize())
print(f"Saved TensorRT engine to {args.out}")
