import os

import numpy as np
import pytest

from audio_editor import AudioEditor
from audio_analysis import integrated_loudness

SR = 44100


@pytest.fixture
def ed():
    return AudioEditor(SR)


def _gain_db(after, before):
    return 20 * np.log10(np.abs(after[:, SR:]).max() / np.abs(before[:, SR:]).max())


def test_peaking_eq_applies_exact_gain_at_centre(ed):
    t = np.arange(SR * 3) / SR
    x = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)
    assert _gain_db(ed.apply_eq(x, [1000], [6.0]), x) == pytest.approx(6.0, abs=0.2)


def test_low_shelf_boosts_below_corner(ed):
    t = np.arange(SR * 3) / SR
    x = np.stack([np.sin(2 * np.pi * 30 * t)] * 2).astype(np.float32)
    y = ed.apply_eq(x, [80, 250, 1000, 4000, 12000], [6, 0, 0, 0, 0], shelf_edges=True)
    assert _gain_db(y, x) == pytest.approx(6.0, abs=0.5)


@pytest.mark.parametrize("room", [0.0, 0.5, 1.0])
def test_reverb_keeps_shape_and_is_finite(ed, stereo_sine, room):
    out = ed.apply_reverb(stereo_sine, room_size=room)
    assert out.shape == stereo_sine.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("ms", [0, 0.001, 50, 1e9])
def test_fades_handle_any_duration(ed, stereo_sine, ms):
    assert ed.fade_in(stereo_sine, ms).shape == stereo_sine.shape
    assert ed.fade_out(stereo_sine, ms).shape == stereo_sine.shape


def test_limiter_respects_ceiling(ed, stereo_sine):
    out = ed.limit(stereo_sine * 50, -1.0)
    assert 20 * np.log10(np.abs(out).max()) <= -0.99


def test_loudness_normalization_hits_target(ed, stereo_sine):
    out = ed.normalize_loudness(stereo_sine, -14.0)
    assert integrated_loudness(out, SR) == pytest.approx(-14.0, abs=0.5)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-3


def test_peak_normalize(ed, stereo_sine):
    out = ed.normalize(stereo_sine, -3.0)
    assert 20 * np.log10(np.abs(out).max()) == pytest.approx(-3.0, abs=0.01)


def test_mix_stems_broadcasts_mono_and_returns_float32(ed, stereo_sine):
    mix = ed.mix_stems({"a": stereo_sine, "b": stereo_sine[:1]}, {"a": 1.0, "b": 0.5})
    assert mix.shape == stereo_sine.shape
    assert mix.dtype == np.float32


def test_mix_stems_zero_weight_is_silent(ed, stereo_sine):
    mix = ed.mix_stems({"a": stereo_sine}, {"a": 0.0})
    assert np.abs(mix).max() == 0


def test_compression_pan_denoise_shapes(ed, stereo_sine):
    assert ed.apply_compression(stereo_sine).shape == stereo_sine.shape
    assert ed.apply_pan(stereo_sine, 0.5).shape == stereo_sine.shape
    assert ed.reduce_noise(stereo_sine[:, :SR]).shape == (2, SR)


def test_time_stretch_slows_down(ed, stereo_sine):
    out = ed.time_stretch(stereo_sine[:, :SR], 0.5)
    assert out.shape[0] == 2 and out.shape[1] > SR * 1.9


def test_pitch_shift_keeps_length(ed, stereo_sine):
    out = ed.pitch_shift(stereo_sine[:, :SR], 2)
    assert out.shape == (2, SR)


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3"])
def test_export_full_length_every_format(ed, tmp_path, fmt):
    # 4 minutes: a single large OGG write used to crash libsndfile
    audio = (np.random.default_rng(0).standard_normal((2, SR * 240)) * 0.1).astype(np.float32)
    path = ed.export_audio(audio, str(tmp_path / f"x.{fmt}"), fmt, "192k")
    assert os.path.getsize(path) > 10_000
