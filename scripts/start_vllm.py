#!/usr/bin/env python3
"""Launch vLLM server with Qwen2.5-VL-7B-Instruct via HF mirror."""
import subprocess
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", "Qwen/Qwen2.5-VL-7B-Instruct",
    "--served-model-name", "qwen2.5-vl-7b",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.55",
    "--dtype", "auto",
    "--enforce-eager",
]
print("HF_ENDPOINT=https://hf-mirror.com")
print("Starting vLLM server:", " ".join(cmd))
subprocess.run(cmd)