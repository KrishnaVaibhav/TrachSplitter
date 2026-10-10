"""Offscreen GUI smoke test: builds the window without loading a model."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path, monkeypatch):
    import config
    import gui
    monkeypatch.setattr(config.Config.__init__, "__defaults__", (str(tmp_path / "cfg.json"),))
    monkeypatch.setattr(gui.TrachSplitterGUI, "load_model", lambda self: None)
    w = gui.TrachSplitterGUI()
    yield w
    w.player.stop()
    w.deleteLater()


def test_presets_respect_available_stems(window, fake_stems):
    import gui
    window.stems = fake_stems
    window.rebuild_mixer(list(fake_stems))
    window.apply_preset("Vocals Only")
    assert set(window._get_mix_weights()) == {"vocals"}
    window.apply_preset("Instrumental")
    assert set(window._get_mix_weights()) == {"drums", "bass", "other"}
    karaoke = {"vocals": fake_stems["vocals"], "no_vocals": fake_stems["drums"]}
    assert gui.preset_active_stems("Instrumental", karaoke) == {"no_vocals"}


def test_render_mix_with_all_processing(window, fake_stems):
    window.stems = fake_stems
    window.rebuild_mixer(list(fake_stems))
    window.eq_sliders[80].setValue(6)
    for check in (window.compression_check, window.reverb_check, window.denoise_check):
        check.setChecked(True)
    window.fade_in_spin.setValue(200)
    window.normalize_mode.setCurrentIndex(2)
    window.pitch_spin.setValue(2)
    window.tempo_spin.setValue(80)
    mix = window._render_mix()
    assert mix.shape[0] == 2 and np.isfinite(mix).all()
    assert np.abs(mix).max() <= 1.0


def test_two_stem_mode_hides_stem_picker(window):
    window.two_stems_combo.setCurrentIndex(window.two_stems_combo.findData("vocals"))
    assert window._separation_choice()[0] == "vocals"
    assert window.stems_group.isHidden()


def test_safe_filename_strips_windows_invalid_chars():
    import gui
    assert gui._safe_filename("Song || Remix| #shorts") == "Song Remix shorts"
    assert gui._safe_filename('a?b*c:d<e>f"g') == "a b c d e f g"
    assert len(gui._safe_filename("x" * 300)) <= 80


def test_menu_exposes_core_commands(window):
    titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
    assert titles == ["File", "Track", "Playback", "View", "Help"]
