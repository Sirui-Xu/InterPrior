"""Discovery and bounded loading of prepared OMOMO motion clips."""

from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Tuple


@dataclass(frozen=True)
class MotionSelection:
    files: Tuple[Path, ...]
    environment_motion_ids: Tuple[int, ...]


def select_motions(directory, num_envs=16, seed=0, motion_file=None, object_name=None):
    """Select at most one motion per environment, allowing unequal object counts."""
    directory = Path(directory).resolve()
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if not directory.is_dir():
        raise FileNotFoundError("Motion directory does not exist: " + str(directory))
    files = sorted(directory.glob("*.pt"))
    if motion_file:
        files = [p for p in files if p.name == Path(motion_file).name]
    if object_name:
        files = [p for p in files if p.stem.split("_")[1:2] == [object_name]]
    if not files:
        raise ValueError("No prepared motion clips match the selection")
    for p in files:
        if not re.fullmatch(r"sub\d+_[A-Za-z0-9]+_.+\.pt", p.name):
            raise ValueError("Expected a prepared subN_object_clip.pt file: " + p.name)
    if len(files) > num_envs:
        files = sorted(random.Random(seed).sample(files, num_envs))
    return MotionSelection(tuple(files), tuple(i % len(files) for i in range(num_envs)))


def load_motion_tensor(path):
    """Read the [frames, 591] SMPL format provided by InterMimic."""
    import torch
    from .serialization import load_tensor_file

    motion = load_tensor_file(path)
    if (
        not isinstance(motion, torch.Tensor)
        or motion.ndim != 2
        or motion.shape[1] != 591
    ):
        raise ValueError(
            "Use InterMimic's teacher-corrected SMPL motions with shape [frames, 591]"
        )
    if len(motion) < 3 or not torch.isfinite(motion).all():
        raise ValueError("Motion needs at least three finite frames")
    return motion.float()
