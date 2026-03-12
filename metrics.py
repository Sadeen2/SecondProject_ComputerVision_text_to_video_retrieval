# utils/io_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(path: Union[str, Path], arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def save_torch(path: Union[str, Path], obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(path))


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    """
    Loads a torch checkpoint safely. (Still uses torch.load; OK if it's your own file.)
    """
    ckpt = torch.load(str(path), map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def build_index_paths(index_dir: Union[str, Path]) -> Dict[str, Path]:
    index_dir = Path(index_dir)
    return {
        "dir": index_dir,
        "meta": index_dir / "meta.json",
        "video_ids": index_dir / "video_ids.json",
        "video_embeds": index_dir / "video_embeddings.npy",
    }


def validate_index_files(index_dir: Union[str, Path]) -> Dict[str, Path]:
    paths = build_index_paths(index_dir)
    for k in ["meta", "video_ids", "video_embeds"]:
        if not paths[k].exists():
            raise FileNotFoundError(f"Missing index file: {paths[k]}")
    return paths
