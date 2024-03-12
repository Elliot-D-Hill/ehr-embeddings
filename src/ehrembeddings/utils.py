from pathlib import Path
from re import search
from numpy import argmin, argmax


# TODO should make it so that if no checkpoint is found, the model is trained
def get_best_checkpoint(ckpt_folder: Path, mode: str) -> Path:
    checkpoint_paths = list(ckpt_folder.glob("*"))
    metrics = [
        float(match.group(1))
        for filepath in checkpoint_paths
        if (match := search(r"(\d+\.\d+)(?=[^\d]|$)", filepath.stem))
    ]
    if mode == "min":
        index = argmin(metrics)
    else:
        index = argmax(metrics)
    return checkpoint_paths[index]
