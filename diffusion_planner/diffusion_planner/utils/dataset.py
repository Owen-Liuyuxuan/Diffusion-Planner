import os
import time
import zipfile

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson
from planner_metrics.temporal_stability import consecutive_frame_pairs

_NPZ_RETRIES = 3
_NPZ_RETRY_EXCEPTIONS = (OSError, zipfile.BadZipFile, ValueError, EOFError)


def _load_npz(path: str) -> dict:
    """Load one observation NPZ, retrying transient USB / CRC read errors."""
    last: BaseException | None = None
    for attempt in range(1, _NPZ_RETRIES + 1):
        try:
            with np.load(path, allow_pickle=True) as loaded:
                data = {key: loaded[key] for key in loaded.files}
            data.pop("version", None)
            return data
        except _NPZ_RETRY_EXCEPTIONS as exc:
            last = exc
            if attempt < _NPZ_RETRIES:
                time.sleep(0.05 * attempt)
                continue
            raise RuntimeError(
                f"failed to load NPZ after {_NPZ_RETRIES} tries: {path}"
            ) from last
    raise RuntimeError(f"failed to load NPZ: {path}") from last


class DiffusionPlannerData(Dataset):
    def __init__(self, data_list):
        if isinstance(data_list, (str, bytes, os.PathLike)):
            self.data_list = openjson(data_list)
        else:
            self.data_list = list(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return _load_npz(self.data_list[idx])


class DiffusionPlannerPairData(Dataset):
    def __init__(self, data_list, expected_gap: int | None = None):
        paths = openjson(data_list)
        expected_gap = expected_gap or None
        self.pairs = list(consecutive_frame_pairs(paths, expected_gap=expected_gap))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, path_a, _, path_b, gap = self.pairs[idx]
        return {
            "current": _load_npz(path_a),
            "next": _load_npz(path_b),
            "frame_gap": np.array(gap, dtype=np.int64),
        }
