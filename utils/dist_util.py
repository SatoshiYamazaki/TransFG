import json
import hashlib
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Tuple, Callable

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0


def detect_device(prefer_mps: bool = True) -> torch.device:
    """Prefer MPS when available, otherwise fall back to CPU (no CUDA dependency).

    CUDA is only used when MPS is unavailable and CUDA exists; this keeps macOS
    runs from assuming CUDA is present.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def map_location_from_device(device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory for torch.load(map_location=...) based on resolved device."""

    def _map(storage: torch.Tensor, loc: str) -> torch.Tensor:  # type: ignore[override]
        return storage.to(device)

    return _map


def set_seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        try:
            import importlib
            importlib.import_module("torch.mps")
        except Exception:
            pass
    torch.use_deterministic_algorithms(False)


def compute_config_hash(args: Mapping[str, Any]) -> str:
    args_json = json.dumps(dict(args), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(args_json).hexdigest()


def requirements_checksum(requirements_path: Path) -> str:
    if not requirements_path.exists():
        return "missing"
    return hashlib.sha256(requirements_path.read_bytes()).hexdigest()


def env_export_hash(env_export_path: Path = Path("env_export.yml")) -> str:
    """Compute a reproducibility hash from env_export.yml or pip freeze fallback."""
    if env_export_path.exists():
        try:
            return hashlib.sha256(env_export_path.read_bytes()).hexdigest()
        except Exception:
            pass
    try:
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], timeout=30)
        return hashlib.sha256(frozen).hexdigest()
    except Exception:
        return "missing"


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def build_env_stamp(args: Any, device: torch.device) -> dict:
    def _safe_version(pkg_name: str) -> str:
        try:
            module = __import__(pkg_name)
            return getattr(module, "__version__", "unknown")
        except Exception:
            return "missing"

    try:
        import torchvision  # pylint: disable=import-outside-toplevel
        tv_version = torchvision.__version__
    except Exception:
        tv_version = "unknown"

    def _torch_build(dev: torch.device) -> str:
        if dev.type == "cuda":
            return f"cuda-{torch.version.cuda or 'unknown'}"
        if dev.type == "mps":
            return "mps"
        return "cpu"

    stamp = {
        "run_id": getattr(args, "name", "unknown"),
        "config_hash": compute_config_hash(vars(args)),
        "git_commit": git_commit_sha(),
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": tv_version,
        "ml_collections_version": _safe_version("ml_collections"),
        "driver_version": "mps" if device.type == "mps" else (torch.version.cuda or "cpu"),
        "cuda_version": torch.version.cuda or "none",
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "none",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("mps" if device.type == "mps" else "cpu"),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else (1 if device.type == "mps" else 0),
        "seed": getattr(args, "seed", 0),
        "requirements_checksum": requirements_checksum(Path("requirements.txt")),
        "env_hash": env_export_hash(),
        "tiny_train_subset_path": getattr(args, "tiny_train_subset", ""),
        "tiny_infer_subset_path": getattr(args, "tiny_infer_subset", ""),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fiftyone_version": _safe_version("fiftyone"),
        "torch_build": _torch_build(device),
        "fp16": False,
    }
    return stamp


def write_env_stamp(args: Any, stamp_path: Path, device: torch.device) -> dict:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = build_env_stamp(args, device)
    stamp_path.write_text(json.dumps(stamp, indent=2))
    logger.info("Wrote env stamp to %s", stamp_path)
    return stamp

def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Training Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Training Iteration: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Iteration: {} ".format(step[2])
    return s
