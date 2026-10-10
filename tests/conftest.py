import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 44100


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep config/log/UI-asset writes out of the real home directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    yield tmp_path


@pytest.fixture
def stereo_sine():
    """5 s, 997 Hz, -20 dBFS per channel stereo sine."""
    t = np.arange(SR * 5) / SR
    s = (10 ** (-20 / 20)) * np.sin(2 * np.pi * 997 * t)
    return np.stack([s, s]).astype(np.float32)


@pytest.fixture
def fake_stems():
    t = np.arange(SR * 3) / SR
    return {
        name: (0.2 * np.stack([np.sin(2 * np.pi * f * t)] * 2)).astype(np.float32)
        for name, f in [("drums", 100), ("bass", 50), ("vocals", 440), ("other", 880)]
    }


def pytest_configure(config):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
