"""Atomic trusted-local checkpoints. Full weights + resumable state, never pickle untrusted files."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import tempfile

import numpy as np
import torch


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)  # Only this newly-created, exact temporary file.


def load_trusted(path, expected_identity=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint")
    if expected_identity is not None and payload.get("identity") != expected_identity:
        raise ValueError("Checkpoint source/data/config/environment identity differs")
    return payload


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    packages = {}
    for name in ("transformers", "scikit-learn", "pandas", "huggingface-hub", "einops"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "packages": packages,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def epoch_order(length, epoch, seed=2021):
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()
