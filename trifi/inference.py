"""
Presence model loading and an off-thread inference worker.

The model is a ResNet-18 with a 1-channel stem, fed one normalised window of
shape (64 subcarriers, window_frames) as an image. Weights are loaded with
torch.load(weights_only=True): plain tensors only, no pickled code.

torch is imported lazily so the rest of the backend (and the test suite) runs
without it; only loading a real model needs torch.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger(__name__)


class Predictor(Protocol):
    """Maps a normalised (n_sub, n_frames) window to class probabilities."""
    n_classes: int

    def predict(self, window: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelInfo:
    path: Path
    n_classes: int

    @property
    def binary(self) -> bool:
        return self.n_classes == 2


class ResNetPresenceModel:
    def __init__(self, path: Path):
        import torch
        import torch.nn as nn
        import torchvision.models as models

        state = torch.load(path, map_location="cpu", weights_only=True)
        fc = state.get("net.fc.weight", state.get("fc.weight"))
        if fc is None:
            raise ValueError(f"{path}: no final layer weights found")
        self.n_classes = int(fc.shape[0])

        class CSICNN(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.net = models.resnet18(num_classes=num_classes)
                self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

            def forward(self, x):
                return self.net(x)

        model = CSICNN(self.n_classes)
        model.load_state_dict(state)
        model.eval()
        self._torch = torch
        self._model = model
        self.path = path

    def predict(self, window: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return torch.softmax(self._model(x), dim=1).numpy()[0]


def find_model(candidates: list[Path]) -> Optional[Path]:
    """First existing model file, in preference order."""
    return next((p for p in candidates if p.is_file()), None)


def load_model(path: Path) -> ResNetPresenceModel:
    model = ResNetPresenceModel(path)
    log.info("Loaded presence model %s (%d classes)", path, model.n_classes)
    return model


# ── Worker process ────────────────────────────────────────────────────────────

_worker_model: Optional[ResNetPresenceModel] = None


def _worker_init(path: str) -> None:
    global _worker_model
    import torch
    torch.set_num_threads(1)
    _worker_model = ResNetPresenceModel(Path(path))


def _worker_predict(window: np.ndarray) -> np.ndarray:
    assert _worker_model is not None
    return _worker_model.predict(window)


class InferenceWorker:
    """
    Runs predictions outside the event loop thread.

    By default a single worker process holds the model (no GIL contention with
    packet intake). Tests can pass any executor and predict function instead.
    """

    def __init__(self, info: ModelInfo,
                 executor: Optional[concurrent.futures.Executor] = None,
                 predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.info = info
        if executor is None:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=1, initializer=_worker_init, initargs=(str(info.path),))
            predict_fn = _worker_predict
        if predict_fn is None:
            raise ValueError("predict_fn is required with a custom executor")
        self._executor = executor
        self._predict = predict_fn

    async def predict(self, window: np.ndarray) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._predict, window)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def probe_model(path: Path) -> ModelInfo:
    """Load once in-process to validate the file and learn its class count."""
    return ModelInfo(path=path, n_classes=load_model(path).n_classes)
