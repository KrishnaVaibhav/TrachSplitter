import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from audio_analysis import integrated_loudness, AudioAnalyzer
from conftest import ROOT, SR


def test_lufs_reference_levels(stereo_sine):
    # BS.1770: -20 dBFS 997 Hz sine on both channels ≈ -20 LUFS, one channel ≈ -23
    assert integrated_loudness(stereo_sine, SR) == pytest.approx(-20.0, abs=0.5)
    assert integrated_loudness(stereo_sine[0], SR) == pytest.approx(-23.0, abs=0.5)


def test_lufs_silence_and_short_input():
    assert integrated_loudness(np.zeros((2, SR * 2)), SR) == float("-inf")
    assert integrated_loudness(np.ones((2, 100)) * 0.1, SR) == float("-inf")


def test_analyze_file_reports_tempo_key_loudness(tmp_path):
    t = np.arange(SR * 8) / SR
    clicks = (np.sin(2 * np.pi * 2 * t) > 0.99) * 0.8  # 120 BPM pulses
    tone = 0.2 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "beat.wav"
    sf.write(path, np.stack([clicks + tone] * 2).T, SR)
    info = AudioAnalyzer().analyze_file(str(path))
    assert info["duration_sec"] == pytest.approx(8.0, abs=0.05)
    assert 50 < info["bpm"] < 250
    assert info["lufs"] is not None and info["lufs"] < 0


def test_separator_loader_resamples_and_upmixes(tmp_path):
    from demucs_separator import DemucsSeparator
    t = np.arange(22050 * 2) / 22050
    path = tmp_path / "mono22k.wav"
    sf.write(path, 0.1 * np.sin(2 * np.pi * 440 * t), 22050)
    wav = DemucsSeparator.load_audio(str(path))
    assert tuple(wav.shape) == (2, SR * 2)


def test_config_roundtrip_and_reset(tmp_path):
    from config import Config
    cfg_path = tmp_path / "cfg.json"
    c = Config(str(cfg_path))
    c.set("output_format", "flac")
    c.save()
    assert Config(str(cfg_path)).get("output_format") == "flac"
    c.reset_to_defaults()
    assert c.get("device") in ("cuda", "mps", "cpu")


@pytest.mark.parametrize("cmd", [[], ["separate"], ["karaoke"], ["analyze"], ["mix"], ["batch"]])
def test_cli_help(cmd):
    r = subprocess.run([sys.executable, str(ROOT / "cli.py"), *cmd, "--help"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def test_cli_mix_end_to_end(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    t = np.arange(SR * 3) / SR
    for name, f in [("drums", 100), ("vocals", 440)]:
        sf.write(stems / f"{name}.wav", (0.2 * np.stack([np.sin(2 * np.pi * f * t)] * 2)).T, SR)
    out = tmp_path / "mix.flac"
    r = subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), "mix", str(stems), "-o", str(out),
         "--mute", "vocals", "-g", "drums:3", "--lufs", "-14"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    data, sr = sf.read(out)
    assert sr == SR and data.shape[1] == 2
