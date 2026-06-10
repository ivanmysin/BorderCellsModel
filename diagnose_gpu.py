"""Diagnose TensorFlow / GPU / CUDA setup.

Run on the server:  python diagnose_gpu.py
"""
import os
import subprocess
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # verbose

print("=" * 60)
print(" 1. Python / venv")
print("=" * 60)
print(f"  python : {sys.executable}")
print(f"  version: {sys.version.split()[0]}")

print()
print("=" * 60)
print(" 2. pip tensorflow")
print("=" * 60)
try:
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "show", "tensorflow"],
        stderr=subprocess.STDOUT).decode()
    for line in out.splitlines():
        if line.startswith(("Name", "Version", "Location", "Requires")):
            print(f"  {line}")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print(" 3. nvidia-smi (driver + GPU presence)")
print("=" * 60)
try:
    out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
    for line in out.splitlines()[:25]:
        print(f"  {line}")
except FileNotFoundError:
    print("  nvidia-smi not on PATH — NVIDIA driver not installed?")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print(" 4. nvcc (CUDA toolkit)")
print("=" * 60)
try:
    out = subprocess.check_output(["nvcc", "--version"],
                                  stderr=subprocess.STDOUT).decode()
    print(out)
except FileNotFoundError:
    print("  nvcc not on PATH — CUDA toolkit not installed or not in PATH.")
    print("  (driver can be present without toolkit, but TF needs both.)")
except Exception as e:
    print(f"  ERROR: {e}")

print()
print("=" * 60)
print(" 5. TF build / GPU visibility")
print("=" * 60)
import tensorflow as tf
print(f"  tf.__version__     : {tf.__version__}")
print(f"  tf.test.is_built_with_cuda() : {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"  tf.config.list_physical_devices('GPU') : {gpus}")
if not gpus:
    print("  >>> NO GPU VISIBLE TO TENSORFLOW <<<")

print()
print("=" * 60)
print(" 6. Try a small matmul on each device")
print("=" * 60)
import numpy as np
a = tf.constant(np.random.randn(1000, 1000).astype(np.float32))
b = tf.constant(np.random.randn(1000, 1000).astype(np.float32))

import time
t0 = time.time()
c_cpu = tf.matmul(a, b)
dt_cpu = time.time() - t0
print(f"  CPU matmul 1000x1000 : {dt_cpu*1000:.1f} ms, result shape {c_cpu.shape}")

if gpus:
    try:
        with tf.device('/GPU:0'):
            t0 = time.time()
            c_gpu = tf.matmul(a, b)
            dt_gpu = time.time() - t0
        print(f"  GPU matmul 1000x1000 : {dt_gpu*1000:.1f} ms, speedup {dt_cpu/dt_gpu:.1f}x")
    except Exception as e:
        print(f"  GPU matmul FAILED: {e}")

print()
print("=" * 60)
print(" 7. Try forcing GPU growth (memory)")
print("=" * 60)
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print("  set_memory_growth OK")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("  no GPUs, skipping")
