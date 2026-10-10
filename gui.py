"""PySide6 GUI for TrachSplitter audio separation and editing."""
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import traceback
from scipy import signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QProgressBar, QTabWidget, QStackedWidget, QDialog,
    QListWidget, QListWidgetItem, QMessageBox, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QGridLayout, QApplication, QStyleFactory
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QUrl, QEvent
from PySide6.QtGui import QIcon, QKeySequence, QAction
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import ui_theme
import subprocess
import time

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif'}
AUDIO_FILTER = "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.aiff *.aif);;All Files (*)"
EXPORT_FILTERS = {
    'wav': "WAV Files (*.wav)",
    'mp3': "MP3 Files (*.mp3)",
    'flac': "FLAC Files (*.flac)",
    'ogg': "OGG Files (*.ogg)",
}
ALL_STEMS = ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']

# Mix presets: ('keep', names) plays only those stems, ('mute', names)
# plays everything except them. Works for 4/6-stem and karaoke results.
MIX_PRESETS = {
    "Full Mix":        ('mute', set()),
    "Instrumental":    ('mute', {'vocals'}),
    "Vocals Only":     ('keep', {'vocals'}),
    "Drums + Bass":    ('keep', {'drums', 'bass'}),
    "Drumless":        ('mute', {'drums'}),
    "Bassless":        ('mute', {'bass'}),
    "Ambient / Other": ('keep', {'other'}),
}


def preset_active_stems(preset: str, stem_names) -> set:
    """Stems that sound under a preset, given the stems we actually have."""
    mode, names = MIX_PRESETS.get(preset, ('mute', set()))
    stem_names = set(stem_names)
    if mode == 'keep':
        return stem_names & names
    return stem_names - names


def _safe_filename(name: str, max_len: int = 80) -> str:
    """Windows-safe file name: drop characters like | # : ? *, collapse runs."""
    import re
    cleaned = "".join(c if c.isalnum() or c in " ._-()&'," else " " for c in name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._-")
    return cleaned[:max_len].rstrip(" ._-") or "track"

from config import Config
from demucs_separator import DemucsSeparator, SeparationCancelled
from audio_editor import AudioEditor
import soundfile as sf
import logging


logger = logging.getLogger(__name__)


class SeparationWorker(QThread):
    """Worker thread for audio separation."""

    progress_update = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, separator: DemucsSeparator, audio_path: str, config: Dict):
        super().__init__()
        self.separator = separator
        self.audio_path = audio_path
        self.config = config

    def cancel(self):
        self.separator.cancel()

    def run(self):
        self.separator.progress_callback = (
            lambda msg, val: self.progress_update.emit(msg, int(val))
        )
        try:
            result = self.separator.separate(
                self.audio_path,
                stem_sources=self.config.get('stem_sources'),
                two_stems=self.config.get('two_stems')
            )
            self.finished.emit(result)
        except SeparationCancelled:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Separation error: {e}\n{traceback.format_exc()}")
        finally:
            self.separator.progress_callback = None


class BatchWorker(QThread):
    """Separate a queue of files, writing each track's stems to its own folder."""

    progress_update = Signal(str, int)
    file_done = Signal(str, bool, str)  # path, ok, message
    all_done = Signal(int, int, bool)   # ok count, failed count, cancelled

    def __init__(self, separator: DemucsSeparator, files: list, out_root: str,
                 fmt: str, bitrate: str, two_stems: Optional[str],
                 stem_sources: Optional[list]):
        super().__init__()
        self.separator = separator
        self.files = files
        self.out_root = out_root
        self.fmt = fmt
        self.bitrate = bitrate
        self.two_stems = two_stems
        self.stem_sources = stem_sources
        self._stop = False

    def cancel(self):
        self._stop = True
        self.separator.cancel()

    def run(self):
        editor = AudioEditor()
        ok = failed = 0
        total = len(self.files)
        for i, path in enumerate(self.files):
            if self._stop:
                break
            name = Path(path).name

            def cb(msg, pct, i=i, name=name):
                overall = int((i + pct / 100.0) / total * 100)
                self.progress_update.emit(f"[{i + 1}/{total}] {name}: {msg}", overall)

            self.separator.progress_callback = cb
            try:
                stems = self.separator.separate(
                    path, stem_sources=self.stem_sources, two_stems=self.two_stems
                )
                out_dir = os.path.join(self.out_root, _safe_filename(Path(path).stem))
                os.makedirs(out_dir, exist_ok=True)
                for stem_name, audio in stems.items():
                    editor.export_audio(
                        audio, os.path.join(out_dir, f"{stem_name}.{self.fmt}"),
                        self.fmt, self.bitrate
                    )
                ok += 1
                self.file_done.emit(path, True, out_dir)
            except SeparationCancelled:
                break
            except Exception as e:
                failed += 1
                logger.error(f"Batch error on {path}: {e}\n{traceback.format_exc()}")
                self.file_done.emit(path, False, str(e))
        self.separator.progress_callback = None
        self.all_done.emit(ok, failed, self._stop)


class TaskWorker(QThread):
    """Generic worker: runs a callable off the UI thread."""

    done = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Task error: {e}\n{traceback.format_exc()}")


class AnalysisWorker(QThread):
    """Worker thread for BPM/key/loudness analysis."""

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, audio_path: str):
        super().__init__()
        self.audio_path = audio_path

    def run(self):
        try:
            from audio_analysis import AudioAnalyzer
            result = AudioAnalyzer().analyze_file(self.audio_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Analysis error: {e}\n{traceback.format_exc()}")


class AudioViewer(QFrame):
    """Waveform + spectrogram of the selected stem, click to seek, drag to select.

    Colours follow the app theme; the waveform takes the stem's identity colour.
    """

    def __init__(self):
        super().__init__()
        self.audio = None
        self.sample_rate = 44100
        self.on_seek = None
        self.on_select = None
        self.duration = 0.0
        self.view_start = 0.0
        self.view_end = None
        self.selection = None
        self.title = ""
        self.wave_color = "#0066CC"
        self.theme = ui_theme.tokens(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.ax_wave = self.figure.add_subplot(211)
        self.ax_spec = self.figure.add_subplot(212, sharex=self.ax_wave)
        self.span = None
        self.playhead = None
        self._spec_data = None
        self._wave_env = None
        self._bg = None
        self._ph_artists = None
        self._last_playhead_draw = 0.0
        self.canvas.mpl_connect("resize_event", lambda e: setattr(self, "_bg", None))
        self.canvas.mpl_connect("button_press_event", self._handle_click)
        layout.addWidget(self.canvas, 1)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Zoom in around the centre")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(0.6))
        toolbar.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(1.6))
        toolbar.addWidget(self.zoom_out_btn)
        self.reset_zoom_btn = QPushButton("Fit")
        self.reset_zoom_btn.setToolTip("Show the whole track")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        toolbar.addWidget(self.reset_zoom_btn)
        toolbar.addSpacing(8)
        self.selection_label = QLabel("Drag on the spectrogram to select a region")
        self.selection_label.setObjectName("caption")
        toolbar.addWidget(self.selection_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.set_theme(self.theme)

    def set_theme(self, theme: dict):
        self.theme = theme
        self.figure.set_facecolor(theme['surface'])
        self.redraw()

    def _style_axes(self):
        t = self.theme
        for ax in (self.ax_wave, self.ax_spec):
            ax.set_facecolor(t['plot_bg'])
            ax.tick_params(colors=t['secondary'], labelsize=8, length=3)
            for spine in ax.spines.values():
                spine.set_color(t['plot_grid'])

    def _handle_click(self, event):
        if self.audio is None or event.inaxes not in (self.ax_wave, self.ax_spec) or event.xdata is None:
            return
        if self.on_seek:
            self.on_seek(float(event.xdata))

    def set_on_seek(self, callback):
        self.on_seek = callback

    def set_on_select(self, callback):
        self.on_select = callback

    @staticmethod
    def compute_view_data(audio: np.ndarray, sample_rate: int = 44100):
        """Compute spectrogram + decimated waveform envelope.

        Heavy — call from a worker thread, then pass the result to
        set_audio_precomputed on the UI thread.
        """
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio
        f, t, sxx = signal.spectrogram(mono, fs=sample_rate, nperseg=1024, noverlap=512)
        spec_data = (f, t, 10 * np.log10(sxx + 1e-10))
        # Min/max envelope: plotting millions of raw samples froze the UI
        n = len(mono)
        hop = max(1, n // 2000)
        trimmed = mono[: (n // hop) * hop].reshape(-1, hop)
        times = (np.arange(trimmed.shape[0]) * hop + hop / 2) / sample_rate
        wave_env = (times, trimmed.min(axis=1), trimmed.max(axis=1))
        duration = n / sample_rate
        return spec_data, wave_env, duration

    def set_audio_precomputed(
        self,
        duration: float,
        sample_rate: int,
        title: str,
        spec_data,
        wave_env,
        color: Optional[str] = None,
    ):
        """Display precomputed view data (cheap, UI-thread safe)."""
        self.audio = True  # sentinel: data loaded
        self.sample_rate = sample_rate
        self.duration = duration
        self.view_start = 0.0
        self.view_end = duration
        self._spec_data = spec_data
        self._wave_env = wave_env
        self.title = title
        if color:
            self.wave_color = color
        self.selection = None
        self.selection_label.setText("Drag on the spectrogram to select a region")
        self.redraw()

    def set_audio(self, audio: np.ndarray, sample_rate: int = 44100, title: str = ""):
        """Synchronous convenience wrapper (blocks — prefer the async path)."""
        spec_data, wave_env, duration = self.compute_view_data(audio, sample_rate)
        self.set_audio_precomputed(duration, sample_rate, title, spec_data, wave_env)

    def clear_audio(self):
        self.audio = None
        self.duration = 0.0
        self.selection = None
        self._spec_data = None
        self._wave_env = None
        self.selection_label.setText("Drag on the spectrogram to select a region")
        self.redraw()

    def update_playhead(self, seconds: Optional[float]):
        """Move (or hide) the playback line via blitting — no full canvas redraw."""
        if not self.audio:
            return
        if seconds is None:
            if self._ph_artists:
                for ln in self._ph_artists:
                    ln.set_visible(False)
            self._bg = None
            self.canvas.draw_idle()
            return
        now = time.monotonic()
        if now - self._last_playhead_draw < 0.05:
            return
        self._last_playhead_draw = now
        if self._bg is None:
            if self._ph_artists is None:
                self._ph_artists = [
                    ax.axvline(0, color=self.theme['content'], linewidth=1.2, animated=True)
                    for ax in (self.ax_wave, self.ax_spec)
                ]
            self.canvas.draw()  # animated artists are excluded → clean background
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
        self.canvas.restore_region(self._bg)
        for ln in self._ph_artists:
            ln.set_xdata([seconds, seconds])
            ln.set_visible(True)
            ln.axes.draw_artist(ln)
        self.canvas.blit(self.figure.bbox)

    def redraw(self, title: Optional[str] = None, playhead_seconds: Optional[float] = None):
        if title:
            self.title = title
        t = self.theme
        self._bg = None
        self._ph_artists = None  # cleared with the axes below
        self.ax_wave.clear()
        self.ax_spec.clear()
        self._style_axes()
        if not self.audio:
            for ax in (self.ax_wave, self.ax_spec):
                ax.set_xticks([])
                ax.set_yticks([])
            self.ax_wave.text(
                0.5, 0.5, "Select a stem to see its waveform",
                ha="center", va="center", color=t['secondary'], fontsize=10,
                transform=self.ax_wave.transAxes,
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        if self._wave_env is not None:
            times, lo, hi = self._wave_env
            self.ax_wave.fill_between(times, lo, hi, color=self.wave_color, linewidth=0)
        self.ax_wave.set_xlim(self.view_start, self.view_end or self.duration)
        self.ax_wave.set_yticks([])
        self.ax_wave.tick_params(labelbottom=False)
        f, tt, sxx_db = self._spec_data
        # Clip to an 80 dB window below the loud end: digital silence sits at
        # -200 dB and would otherwise push all real content into the top colours
        vmax = float(np.percentile(sxx_db, 99.5)) if sxx_db.size else 0.0
        self.ax_spec.pcolormesh(tt, f, sxx_db, shading="auto", cmap="magma",
                                vmin=vmax - 80.0, vmax=vmax, rasterized=True)
        self.ax_spec.set_ylim(0, min(16000, self.sample_rate // 2))
        self.ax_spec.set_xlim(self.view_start, self.view_end or self.duration)
        self.ax_spec.set_xlabel("Time (s)", color=t['secondary'], fontsize=9)
        self.ax_spec.set_ylabel("Hz", color=t['secondary'], fontsize=9)
        if self.span is None:
            self.span = SpanSelector(
                self.ax_spec,
                self._handle_span,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.25, facecolor=t['accent'])
            )
        if playhead_seconds is not None:
            for ax in (self.ax_wave, self.ax_spec):
                ax.axvline(playhead_seconds, color=t['content'], linewidth=1.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _handle_span(self, xmin, xmax):
        self.selection = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        self.selection_label.setText(
            f"Selected {self.selection[0]:.2f}s – {self.selection[1]:.2f}s "
            f"({self.selection[1] - self.selection[0]:.2f}s)"
        )
        if self.on_select:
            self.on_select(*self.selection)

    def zoom(self, factor: float):
        if self.audio is None:
            return
        center = (self.view_start + (self.view_end or self.duration)) / 2
        span = ((self.view_end or self.duration) - self.view_start) * factor
        span = max(1.0, min(span, self.duration))
        self.view_start = max(0.0, center - span / 2)
        self.view_end = min(self.duration, center + span / 2)
        self.redraw()

    def reset_zoom(self):
        if self.audio is None:
            return
        self.view_start = 0.0
        self.view_end = self.duration
        self.redraw()


class ModelLoadWorker(QThread):
    """Worker thread for loading/switching Demucs models."""

    finished = Signal(object)
    error = Signal(str)

    def __init__(self, model_name: str, device: Optional[str]):
        super().__init__()
        self.model_name = model_name
        self.device = device

    def run(self):
        try:
            sep = DemucsSeparator(
                model_name=self.model_name,
                device=self.device,
                segment=None,
                overlap=0.25,
                shifts=1
            )
            self.finished.emit(sep)
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Model load error: {e}\n{traceback.format_exc()}")


class TrachSplitterGUI(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.separator: Optional[DemucsSeparator] = None
        self.stems: Dict[str, np.ndarray] = {}
        self.editor = AudioEditor()
        self.working_directory = self.config.get('working_directory') or str(
            Path.home() / "Music" / "TrachSplitter"
        )
        self.preview_temp_dir = None
        self.preview_files: Dict[str, str] = {}
        self.current_preview_stem: Optional[str] = None
        self._mix_temp_dir: Optional[str] = None
        self.stem_mutes: Dict[str, object] = {}
        self.stem_solos: Dict[str, object] = {}
        self.stem_pans: Dict[str, QSlider] = {}
        self.recent_files: list = [
            p for p in self.config.get('recent_files', []) if os.path.exists(p)
        ]

        self._task_workers: list = []
        self._render_buttons: list = []
        self._model_workers: list = []  # keep refs: a GC'd running QThread crashes
        self.worker: Optional[SeparationWorker] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._analysis_result: Optional[dict] = None
        self.mixer_rows: Dict[str, QWidget] = {}
        self.stem_sliders: Dict[str, QSlider] = {}
        self.stem_labels: Dict[str, QLabel] = {}

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_player_position)
        self._playing_tag: Optional[str] = None

        self.setAcceptDrops(True)

        self.init_ui()
        self._init_shortcuts()
        self.load_model()

    def save_current(self):
        """Ctrl+S: save the configured mix on the mixing tab, else the stems."""
        if self.tabs.currentIndex() == 1:
            self.export_mix()
        else:
            self.save_stems()

    def toggle_playback(self):
        """Space: stop if playing, else play the current context."""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.stop_playback()
        elif self.tabs.currentIndex() == 1:
            self.preview_mix()
        else:
            self.play_selected_stem()

    def _is_busy(self) -> bool:
        return bool(
            (self.worker and self.worker.isRunning())
            or (self._batch_worker and self._batch_worker.isRunning())
        )

    # ── Background tasks ────────────────────────────────────────────────────

    def _run_task(self, fn, on_done, on_error=None):
        """Run fn on a worker thread; deliver result/error on the UI thread."""
        worker = TaskWorker(fn)
        self._task_workers.append(worker)
        worker.done.connect(on_done)
        worker.error.connect(
            on_error or (lambda msg: QMessageBox.critical(self, "Error", msg))
        )

        def _cleanup():
            if worker in self._task_workers:
                self._task_workers.remove(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup)
        worker.start()

    def _set_rendering(self, busy: bool, message: str = ""):
        """Toggle render-related buttons while a mix renders off-thread."""
        for btn in self._render_buttons:
            btn.setEnabled(not busy)
        if busy:
            self.statusBar().showMessage(message or "Rendering…")

    # ── Drag & drop ─────────────────────────────────────────────────────────

    @staticmethod
    def _audio_paths_from_urls(urls) -> list:
        """Audio files from dropped URLs; dropped folders are scanned."""
        paths = []
        for url in urls:
            p = Path(url.toLocalFile())
            if p.is_dir():
                paths += sorted(
                    str(f) for f in p.iterdir()
                    if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
                )
            elif p.suffix.lower() in AUDIO_EXTENSIONS:
                paths.append(str(p))
        return paths

    def _set_drop_highlight(self, on: bool):
        zone = getattr(self, 'drop_zone', None)
        if zone is not None:
            zone.setProperty("active", "true" if on else "false")
            zone.style().unpolish(zone)
            zone.style().polish(zone)

    def dragEnterEvent(self, event):
        if self._audio_paths_from_urls(event.mimeData().urls()):
            event.acceptProposedAction()
            self._set_drop_highlight(True)

    def dragLeaveEvent(self, event):
        self._set_drop_highlight(False)

    def dropEvent(self, event):
        self._set_drop_highlight(False)
        paths = self._audio_paths_from_urls(event.mimeData().urls())
        if not paths:
            return
        event.acceptProposedAction()
        if len(paths) == 1:
            self.load_audio_file(paths[0])
        else:
            self.start_batch(paths)

    # ── UI construction ─────────────────────────────────────────────────────
    #
    # Layout: sidebar (navigation + recent files) | page stack.
    # Every command is in the menu bar; settings live in their own window;
    # results are reported inline (banner), not with modal alerts.

    PAGE_SEPARATE, PAGE_MIX = 0, 1

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("TrachSplitter")
        for icon_name in ("icon.ico", "icon.svg"):
            icon_path = Path(__file__).resolve().parent / icon_name
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                break
        self.setMinimumSize(1024, 680)
        saved_geo = self.config.get('window_geometry')
        screen = QApplication.primaryScreen().availableGeometry()
        if (isinstance(saved_geo, list) and len(saved_geo) == 4
                and saved_geo[2] >= 1024 and saved_geo[3] >= 680
                and saved_geo[0] < screen.right() and saved_geo[1] < screen.bottom()):
            self.setGeometry(*saved_geo)
        else:
            w = min(1440, int(screen.width() * 0.85))
            h = min(920, int(screen.height() * 0.88))
            self.setGeometry(
                screen.x() + (screen.width() - w) // 2,
                screen.y() + (screen.height() - h) // 2,
                w, h
            )

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())

        main_col = QWidget()
        main_col.setObjectName("page")
        col = QVBoxLayout(main_col)
        col.setContentsMargins(20, 16, 20, 12)
        col.setSpacing(10)
        col.addWidget(self._build_banner())

        self.tabs = QStackedWidget()
        self.tabs.addWidget(self.create_separation_tab())
        self.tabs.addWidget(self.create_editing_tab())
        col.addWidget(self.tabs, 1)
        root.addWidget(main_col, 1)

        self.settings_dialog = self._build_settings_dialog()
        self.nav_list.setCurrentRow(self.PAGE_SEPARATE)
        self.statusBar().showMessage("Ready")
        self.apply_theme()

    # ── Theme ───────────────────────────────────────────────────────────────

    def apply_theme(self):
        """Apply the light or dark tokens that match the system appearance."""
        self.dark = ui_theme.system_is_dark()
        t = ui_theme.tokens(self.dark)
        app = QApplication.instance()
        ui_theme.apply_palette(app, t)
        app.setStyleSheet(ui_theme.stylesheet(t))
        self.spectrogram_view.set_theme(t)
        # Stem colours have light/dark variants: refresh every identity mark
        for i in range(self.stem_output_list.count()):
            item = self.stem_output_list.item(i)
            item.setIcon(ui_theme.dot_icon(ui_theme.stem_color(item.data(Qt.ItemDataRole.UserRole), self.dark)))
        for stem, cb in self.stem_checkboxes.items():
            cb.setIcon(ui_theme.dot_icon(ui_theme.stem_color(stem, self.dark)))
        for stem, lbl in getattr(self, 'mixer_dots', {}).items():
            lbl.setPixmap(ui_theme.dot_icon(ui_theme.stem_color(stem, self.dark)).pixmap(12, 12))

    def changeEvent(self, event):
        # Windows/macOS send PaletteChange/ThemeChange when the appearance flips
        if event.type() in (QEvent.Type.ThemeChange, QEvent.Type.ApplicationPaletteChange):
            if hasattr(self, 'spectrogram_view') and getattr(self, 'dark', None) is not None:
                if ui_theme.system_is_dark() != self.dark:
                    QTimer.singleShot(0, self.apply_theme)
        super().changeEvent(event)

    def get_stylesheet(self) -> str:
        """Current theme stylesheet (kept for API compatibility)."""
        return ui_theme.stylesheet(ui_theme.tokens(getattr(self, 'dark', False)))

    # ── Building blocks ─────────────────────────────────────────────────────

    @staticmethod
    def _card(title: Optional[str] = None, caption: Optional[str] = None):
        """Rounded content group; returns (frame, body layout)."""
        frame = QFrame()
        frame.setObjectName("card")
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(10)
        if title:
            head = QLabel(title)
            head.setObjectName("section")
            lay.addWidget(head)
        if caption:
            cap = QLabel(caption)
            cap.setObjectName("secondary")
            cap.setWordWrap(True)
            lay.addWidget(cap)
        return frame, lay

    @staticmethod
    def _form_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("formLabel")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setObjectName("hline")
        return line

    def _page_header(self, title: str, subtitle_attr: Optional[str] = None):
        """Page title on the left, page actions on the right."""
        row = QHBoxLayout()
        row.setSpacing(8)
        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        lbl = QLabel(title)
        lbl.setObjectName("title")
        text_col.addWidget(lbl)
        if subtitle_attr:
            sub = QLabel("")
            sub.setObjectName("secondary")
            setattr(self, subtitle_attr, sub)
            text_col.addWidget(sub)
        row.addLayout(text_col)
        row.addStretch()
        return row

    # ── Sidebar ─────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(208)
        lay = QVBoxLayout(side)
        lay.setContentsMargins(0, 12, 0, 12)
        lay.setSpacing(0)

        brand = QLabel("TrachSplitter")
        brand.setObjectName("section")
        brand.setContentsMargins(18, 0, 0, 8)
        lay.addWidget(brand)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("nav")
        self.nav_list.setAccessibleName("Sections")
        for text, tip in (("Separate", "Split a song into stems (Ctrl+1)"),
                          ("Mix", "Balance stems, add effects and export (Ctrl+2)")):
            item = QListWidgetItem(text)
            item.setToolTip(tip)
            self.nav_list.addItem(item)
        self.nav_list.setFixedHeight(2 * 34 + 8)
        self.nav_list.currentRowChanged.connect(self._on_nav_changed)
        lay.addWidget(self.nav_list)

        hdr = QLabel("RECENT")
        hdr.setObjectName("sidebarHeader")
        lay.addWidget(hdr)
        self.recent_list = QListWidget()
        self.recent_list.setObjectName("recent")
        self.recent_list.setAccessibleName("Recent files")
        self.recent_list.itemActivated.connect(self._on_recent_item)
        self.recent_list.itemClicked.connect(self._on_recent_item)
        lay.addWidget(self.recent_list, 1)
        self._refresh_recent_combo()
        return side

    def _on_nav_changed(self, row: int):
        if row >= 0:
            self.tabs.setCurrentIndex(row)

    def show_page(self, index: int):
        self.nav_list.setCurrentRow(index)

    # ── Inline notifications ────────────────────────────────────────────────

    def _build_banner(self) -> QFrame:
        self.banner = QFrame()
        self.banner.setObjectName("banner")
        lay = QHBoxLayout(self.banner)
        lay.setContentsMargins(12, 6, 6, 6)
        lay.setSpacing(8)
        self.banner_label = QLabel("")
        self.banner_label.setWordWrap(True)
        lay.addWidget(self.banner_label, 1)
        self.banner_action = QPushButton("Show in Folder")
        self.banner_action.setObjectName("link_btn")
        self.banner_action.clicked.connect(self._banner_open)
        lay.addWidget(self.banner_action)
        close = QPushButton("Dismiss")
        close.setObjectName("link_btn")
        close.clicked.connect(self.banner.hide)
        lay.addWidget(close)
        self.banner.hide()
        self._banner_path = None
        self._banner_timer = QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self.banner.hide)
        return self.banner

    def notify(self, message: str, path: Optional[str] = None, timeout_ms: int = 10000):
        """Non-modal result message with an optional 'Show in Folder' action."""
        self.banner_label.setText(message)
        self._banner_path = path
        self.banner_action.setVisible(bool(path))
        self.banner.show()
        self.statusBar().showMessage(message, 8000)
        self._banner_timer.start(timeout_ms)

    def _banner_open(self):
        if not self._banner_path:
            return
        target = self._banner_path
        if os.path.isfile(target):
            if sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", os.path.normpath(target)])
                return
            target = os.path.dirname(target)
        self._open_path(target)

    # ── Menu bar: every command lives here ──────────────────────────────────

    def _init_shortcuts(self):
        """Build the menu bar; menu shortcuts double as keyboard shortcuts."""
        mb = self.menuBar()
        mb.clear()

        def act(menu, text, slot, shortcut=None, tip=None, role=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setStatusTip(tip)
            if role is not None:
                a.setMenuRole(role)
            a.triggered.connect(lambda checked=False: slot())
            menu.addAction(a)
            return a

        m_file = mb.addMenu("&File")
        act(m_file, "&Open…", self.browse_audio_file, "Ctrl+O", "Open an audio file")
        self.recent_menu = m_file.addMenu("Open &Recent")
        self._refresh_recent_menu()
        act(m_file, "&Batch Separate…", lambda: self.start_batch(), "Ctrl+B",
            "Separate many files and save each track's stems")
        m_file.addSeparator()
        act(m_file, "&Save All Stems…", self.save_stems, "Ctrl+Shift+S")
        act(m_file, "&Export My Mix…", self.export_mix, "Ctrl+E",
            "Save the mix as configured on the Mix page")
        act(m_file, "Save (Mix or Stems)", self.save_current, "Ctrl+S")
        m_file.addSeparator()
        act(m_file, "Se&ttings…", self.open_settings, "Ctrl+,",
            role=QAction.MenuRole.PreferencesRole)
        m_file.addSeparator()
        act(m_file, "E&xit", self.close, "Ctrl+Q", role=QAction.MenuRole.QuitRole)

        m_track = mb.addMenu("&Track")
        act(m_track, "&Separate", self.start_separation, "Ctrl+Return")
        act(m_track, "&Cancel Separation", self.cancel_separation, "Escape")
        m_track.addSeparator()
        act(m_track, "&Analyze Tempo && Key", self.analyze_track, "Ctrl+I")
        act(m_track, "Export &Click Track…", self.export_click_track)

        m_play = mb.addMenu("&Playback")
        act(m_play, "&Play / Stop", self.toggle_playback, "Space")
        act(m_play, "Play Selected &Stem", self.play_selected_stem)
        act(m_play, "Preview &Mix", self.preview_mix, "Ctrl+P")
        act(m_play, "S&top", self.stop_playback, "Ctrl+.")

        m_view = mb.addMenu("&View")
        act(m_view, "&Separate", lambda: self.show_page(self.PAGE_SEPARATE), "Ctrl+1")
        act(m_view, "&Mix", lambda: self.show_page(self.PAGE_MIX), "Ctrl+2")

        m_help = mb.addMenu("&Help")
        act(m_help, "&Keyboard Shortcuts", self.show_shortcuts, "F1")
        act(m_help, "Show &Log Folder", lambda: self._open_path(str(Path.home() / '.trachsplitter')))
        act(m_help, "&About TrachSplitter", self.show_about, role=QAction.MenuRole.AboutRole)

    def show_shortcuts(self):
        rows = [
            ("Open file", "Ctrl+O"), ("Separate", "Ctrl+Enter"), ("Cancel", "Esc"),
            ("Batch separate", "Ctrl+B"), ("Play / stop", "Space"), ("Preview mix", "Ctrl+P"),
            ("Save mix (Mix page) or stems", "Ctrl+S"), ("Export my mix", "Ctrl+E"),
            ("Save all stems", "Ctrl+Shift+S"), ("Analyze", "Ctrl+I"),
            ("Separate / Mix page", "Ctrl+1 / Ctrl+2"), ("Settings", "Ctrl+,"),
        ]
        html = "<table cellspacing='6'>" + "".join(
            f"<tr><td>{a}</td><td><b>{k}</b></td></tr>" for a, k in rows
        ) + "</table>"
        QMessageBox.information(self, "Keyboard Shortcuts", html)

    def show_about(self):
        dev = self.separator.device.upper() if self.separator else "not loaded"
        QMessageBox.about(
            self, "About TrachSplitter",
            "<b>TrachSplitter</b><br>AI stem separation and mixing.<br><br>"
            f"Separation by Meta's Demucs v4 · running on {dev}.<br>"
            "Everything runs locally; your audio never leaves this computer."
        )

    # ── Separate page ───────────────────────────────────────────────────────

    def create_separation_tab(self) -> QWidget:
        """Separate page: source → options → separate (left), stems (right)."""
        page = QWidget()
        page.setObjectName("page")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        header = self._page_header("Separate")
        self.batch_btn = QPushButton("Batch Separate…")
        self.batch_btn.setToolTip(
            "Pick many files (or drop several / a folder on the window).\n"
            "Each track's stems are saved to its own folder. (Ctrl+B)"
        )
        self.batch_btn.clicked.connect(lambda: self.start_batch())
        header.addWidget(self.batch_btn)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        outer.addWidget(splitter, 1)

        # ── Left column ─────────────────────────────────────────────────────
        left = QWidget()
        left.setObjectName("page")
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(12)

        # Source: empty-state drop zone, or the loaded track
        self.source_stack = QStackedWidget()
        drop = QFrame()
        drop.setObjectName("dropZone")
        drop_lay = QVBoxLayout(drop)
        drop_lay.setContentsMargins(20, 24, 20, 24)
        drop_lay.setSpacing(8)
        drop_title = QLabel("Drop a song here")
        drop_title.setObjectName("section")
        drop_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_lay.addWidget(drop_title)
        drop_cap = QLabel("MP3, WAV, FLAC, OGG, M4A · drop a folder to batch")
        drop_cap.setObjectName("secondary")
        drop_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_lay.addWidget(drop_cap)
        open_btn = QPushButton("Open…")
        open_btn.setObjectName("primary_btn")
        open_btn.setToolTip("Ctrl+O")
        open_btn.clicked.connect(self.browse_audio_file)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(open_btn)
        row.addStretch()
        drop_lay.addLayout(row)
        self.drop_zone = drop
        self.source_stack.addWidget(drop)

        track_card, track_lay = self._card()
        self.file_path_label = QLabel("")
        self.file_path_label.setObjectName("file_loaded")
        self.file_path_label.setWordWrap(True)
        track_lay.addWidget(self.file_path_label)
        self.analysis_label = QLabel("Tempo, key and loudness: not analyzed yet")
        self.analysis_label.setObjectName("analysis_result")
        self.analysis_label.setWordWrap(True)
        track_lay.addWidget(self.analysis_label)
        track_btns = QHBoxLayout()
        track_btns.setSpacing(8)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setToolTip("Detect tempo, key and loudness (Ctrl+I)")
        self.analyze_btn.clicked.connect(self.analyze_track)
        track_btns.addWidget(self.analyze_btn)
        self.click_btn = QPushButton("Click Track…")
        self.click_btn.setToolTip("Save a metronome aligned to the detected beats")
        self.click_btn.setEnabled(False)
        self.click_btn.clicked.connect(self.export_click_track)
        track_btns.addWidget(self.click_btn)
        track_btns.addStretch()
        change_btn = QPushButton("Change…")
        change_btn.clicked.connect(self.browse_audio_file)
        track_btns.addWidget(change_btn)
        track_lay.addLayout(track_btns)
        self.source_stack.addWidget(track_card)
        self.source_stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.source_stack)

        # Options
        opt_card, opt_lay = self._card("Options")
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)

        form.addWidget(self._form_label("Model"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.setAccessibleName("Separation model")
        for group, model_ids in self.config.MODEL_GROUPS.items():
            self.model_combo.addItem(f"{group}", userData=None)
            idx = self.model_combo.count() - 1
            self.model_combo.model().item(idx).setEnabled(False)
            for model_id in model_ids:
                short = self.config.MODEL_NOTES.get(model_id, model_id)
                self.model_combo.addItem(f"   {model_id} — {short}", userData=model_id)
        self.model_combo.setToolTip(
            "Best Quality: slower, cleanest stems\n"
            "Balanced: good speed/quality trade-off\n"
            "Fast / Light: quickest, lowest memory\n"
            "htdemucs_6s adds guitar and piano"
        )
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.set_model_combo(self.config.get('model', 'htdemucs'))
        form.addWidget(self.model_combo, 0, 1)
        self.model_info_label = QLabel("")
        self.model_info_label.setObjectName("caption")
        self.model_info_label.setWordWrap(True)
        form.addWidget(self.model_info_label, 1, 1)

        form.addWidget(self._form_label("Output"), 2, 0)
        self.two_stems_combo = QComboBox()
        self.two_stems_combo.setAccessibleName("Output mode")
        self.two_stems_combo.addItem("All stems", userData=None)
        self.two_stems_combo.setToolTip(
            "All stems: one file per instrument.\n"
            "'Vocals + everything else' gives an acapella and a karaoke backing track."
        )
        self.two_stems_combo.currentIndexChanged.connect(self._on_output_mode_changed)
        form.addWidget(self.two_stems_combo, 2, 1)

        self.stems_label = self._form_label("Stems")
        self.stems_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        form.addWidget(self.stems_label, 3, 0)
        self.stems_group = QWidget()
        self.stems_layout = QGridLayout(self.stems_group)
        self.stems_layout.setContentsMargins(0, 0, 0, 0)
        self.stems_layout.setHorizontalSpacing(14)
        self.stems_layout.setVerticalSpacing(4)
        self.stem_checkboxes = {}
        for i, stem in enumerate(ALL_STEMS):
            checkbox = QCheckBox(stem.capitalize())
            checkbox.setChecked(stem in ['drums', 'bass', 'vocals', 'other'])
            self.stem_checkboxes[stem] = checkbox
            self.stems_layout.addWidget(checkbox, i // 3, i % 3)
        form.addWidget(self.stems_group, 3, 1)
        self.model_caps_label = QLabel("")
        self.model_caps_label.hide()  # stem availability is shown by the checkboxes
        self.two_stems_group = self.two_stems_combo

        # Advanced: progressive disclosure
        self.advanced_btn = QPushButton("▸ Advanced")
        self.advanced_btn.setObjectName("link_btn")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        form.addWidget(self.advanced_btn, 4, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        self.advanced_box = QWidget()
        adv = QHBoxLayout(self.advanced_box)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(8)
        adv.addWidget(QLabel("Segment"))
        self.segment_spin = QSpinBox()
        self.segment_spin.setRange(0, 60)
        self.segment_spin.setSpecialValueText("Auto")
        self.segment_spin.setSuffix(" s")
        self.segment_spin.setToolTip("Chunk length. Lower it if your GPU runs out of memory.")
        adv.addWidget(self.segment_spin)
        adv.addSpacing(8)
        adv.addWidget(QLabel("Passes"))
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setRange(1, 10)
        self.shifts_spin.setToolTip("Prediction passes. More = slightly cleaner, proportionally slower.")
        adv.addWidget(self.shifts_spin)
        adv.addStretch()
        self.advanced_box.hide()
        form.addWidget(self.advanced_box, 5, 1)
        opt_lay.addLayout(form)

        self.auto_save_check = QCheckBox("Save stems to the working folder automatically")
        self.auto_save_check.setToolTip("Writes every stem to <working folder>/<track name>/")
        self.auto_save_check.setChecked(bool(self.config.get('auto_save_stems', False)))
        self.auto_save_check.toggled.connect(lambda v: self._save_setting('auto_save_stems', v))
        opt_lay.addWidget(self.auto_save_check)
        layout.addWidget(opt_card)

        # Primary action + progress
        act_row = QHBoxLayout()
        act_row.setSpacing(8)
        self.separate_btn = QPushButton("Separate")
        self.separate_btn.setObjectName("primary_btn")
        self.separate_btn.setToolTip("Ctrl+Enter")
        self.separate_btn.setMinimumHeight(34)
        self.separate_btn.clicked.connect(self.start_separation)
        act_row.addWidget(self.separate_btn, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Esc")
        self.cancel_btn.setMinimumHeight(34)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_separation)
        act_row.addWidget(self.cancel_btn)
        layout.addLayout(act_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setAccessibleName("Separation progress")
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("secondary")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(480)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        # ── Right column: stems ─────────────────────────────────────────────
        res_card, res_lay = self._card()
        res_head = QHBoxLayout()
        res_title = QLabel("Stems")
        res_title.setObjectName("section")
        res_head.addWidget(res_title)
        res_head.addStretch()
        self.save_all_btn = QPushButton("Save All Stems…")
        self.save_all_btn.setToolTip("Write each stem as its own file (Ctrl+Shift+S)")
        self.save_all_btn.clicked.connect(self.save_stems)
        self._render_buttons.append(self.save_all_btn)
        res_head.addWidget(self.save_all_btn)
        res_lay.addLayout(res_head)

        body = QHBoxLayout()
        body.setSpacing(12)
        list_col = QVBoxLayout()
        list_col.setSpacing(6)
        self.stem_output_list = QListWidget()
        self.stem_output_list.setAccessibleName("Separated stems")
        self.stem_output_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.stem_output_list.setToolTip(
            "Ctrl/Shift-click to select several stems:\n"
            "they play together and merge into one file."
        )
        self.stem_output_list.setIconSize(QSize(12, 12))
        self.stem_output_list.setFixedWidth(200)
        self.stem_output_list.itemSelectionChanged.connect(self.on_stem_selection_changed)
        list_col.addWidget(self.stem_output_list, 1)
        self.stems_empty_label = QLabel("Stems appear here after separating.")
        self.stems_empty_label.setObjectName("caption")
        self.stems_empty_label.setWordWrap(True)
        self.stems_empty_label.setFixedWidth(200)
        list_col.addWidget(self.stems_empty_label)
        body.addLayout(list_col)

        viewer_col = QVBoxLayout()
        viewer_col.setSpacing(8)
        self.spectrogram_view = AudioViewer()
        self.spectrogram_view.set_on_select(self.on_spectrogram_selection)
        self.spectrogram_view.set_on_seek(self.seek_playback)
        self.spectrogram_view.setMinimumHeight(260)
        viewer_col.addWidget(self.spectrogram_view, 1)

        transport = QHBoxLayout()
        transport.setSpacing(8)
        play_stem_btn = QPushButton("▶  Play")
        play_stem_btn.setToolTip("Play the selected stem(s). Click the waveform to seek. (Space)")
        play_stem_btn.clicked.connect(self.play_selected_stem)
        transport.addWidget(play_stem_btn)
        stop_stem_btn = QPushButton("■  Stop")
        stop_stem_btn.clicked.connect(self.stop_playback)
        transport.addWidget(stop_stem_btn)
        transport.addStretch()
        export_sel_btn = QPushButton("Export Selection…")
        export_sel_btn.setToolTip("Save the region dragged on the spectrogram")
        export_sel_btn.clicked.connect(self.export_selection)
        self._render_buttons.append(export_sel_btn)
        transport.addWidget(export_sel_btn)
        open_btn2 = QPushButton("Open in Player")
        open_btn2.setToolTip("Open the stem in your default audio player")
        open_btn2.clicked.connect(self.open_stem_in_player)
        transport.addWidget(open_btn2)
        viewer_col.addLayout(transport)
        body.addLayout(viewer_col, 1)
        res_lay.addLayout(body, 1)

        res_lay.addWidget(self._hline())
        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        export_row.addWidget(self._form_label("Quick export"))
        self.quick_preset_combo = QComboBox()
        self.quick_preset_combo.setAccessibleName("Quick export preset")
        self.quick_preset_combo.addItems(list(MIX_PRESETS.keys()))
        self.quick_preset_combo.setToolTip(
            "Exports straight from the raw stems (ignores the Mix page).\n"
            "Instrumental = no vocals · Vocals Only = acapella ·\n"
            "Drumless / Bassless = play-along tracks"
        )
        export_row.addWidget(self.quick_preset_combo)
        quick_export_btn = QPushButton("Export Preset…")
        quick_export_btn.clicked.connect(self.quick_export_mix)
        self._render_buttons.append(quick_export_btn)
        export_row.addWidget(quick_export_btn)
        export_row.addStretch()
        merge_btn = QPushButton("Merge Selected…")
        merge_btn.setToolTip("Mix the stems selected in the list into one file")
        merge_btn.clicked.connect(self.export_merged_selection)
        self._render_buttons.append(merge_btn)
        export_row.addWidget(merge_btn)
        res_lay.addLayout(export_row)

        splitter.addWidget(res_card)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 900])

        self.sync_stem_controls()
        return page

    def _toggle_advanced(self, shown: bool):
        self.advanced_box.setVisible(shown)
        self.advanced_btn.setText("▾ Advanced" if shown else "▸ Advanced")

    def _on_output_mode_changed(self, _index: int = 0):
        two = self.two_stems_combo.currentData() is not None
        self.stems_group.setVisible(not two)
        self.stems_label.setVisible(not two)

    # ── Mix page ────────────────────────────────────────────────────────────

    def create_editing_tab(self) -> QWidget:
        """Mix page: channel rows left, processing inspector right,
        transport + export in the page header (never in a bottom bar)."""
        page = QWidget()
        page.setObjectName("page")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        header = self._page_header("Mix", subtitle_attr="mix_subtitle")
        self.mix_subtitle.setText("Separate a track first, then balance its stems here.")
        play_mix_btn = QPushButton("▶  Preview")
        play_mix_btn.setToolTip("Render and play the mix as configured (Ctrl+P / Space)")
        play_mix_btn.clicked.connect(self.preview_mix)
        self._render_buttons.append(play_mix_btn)
        header.addWidget(play_mix_btn)
        stop_mix_btn = QPushButton("■  Stop")
        stop_mix_btn.clicked.connect(self.stop_preview)
        header.addWidget(stop_mix_btn)
        export_btn = QPushButton("Export My Mix…")
        export_btn.setObjectName("primary_btn")
        export_btn.setToolTip(
            "Save the mix exactly as configured: volumes, mute/solo, pan,\n"
            "EQ, effects, pitch/speed and loudness (Ctrl+E)"
        )
        export_btn.clicked.connect(self.export_mix)
        self._render_buttons.append(export_btn)
        header.addWidget(export_btn)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        outer.addWidget(splitter, 1)

        # ── Mixer ───────────────────────────────────────────────────────────
        left = QWidget()
        left.setObjectName("page")
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(12)

        mixer_card, mixer_lay = self._card("Stems")
        head = QHBoxLayout()
        head.setSpacing(8)
        for text, width, stretch in (("", 12, 0), ("Stem", 0, 2), ("Mute", 28, 0), ("Solo", 28, 0),
                                     ("Volume", 0, 5), ("", 52, 0), ("Pan", 0, 2)):
            lbl = QLabel(text)
            lbl.setObjectName("caption")
            if width:
                lbl.setFixedWidth(width)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                head.addWidget(lbl)
            else:
                head.addWidget(lbl, stretch)
        mixer_lay.addLayout(head)
        self.mixer_rows_layout = QVBoxLayout()
        self.mixer_rows_layout.setSpacing(4)
        mixer_lay.addLayout(self.mixer_rows_layout)
        self.mixer_empty_label = QLabel("No stems yet. Separate a track on the Separate page.")
        self.mixer_empty_label.setObjectName("secondary")
        mixer_lay.addWidget(self.mixer_empty_label)
        layout.addWidget(mixer_card)

        presets_card, presets_lay = self._card(
            "Presets", "Resets volumes and pans, then mutes stems to match."
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for i, name in enumerate(MIX_PRESETS):
            btn = QPushButton(name)
            btn.setObjectName("preset_btn")
            btn.clicked.connect(lambda checked, n=name: self.apply_preset(n))
            grid.addWidget(btn, i // 4, i % 4)
        presets_lay.addLayout(grid)
        layout.addWidget(presets_card)
        layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(480)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        # ── Inspector: processing in signal-flow order ──────────────────────
        right = QWidget()
        right.setObjectName("page")
        layout = QVBoxLayout(right)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(12)

        eq_card, eq_lay = self._card("Tone")
        eq_grid = QGridLayout()
        eq_grid.setHorizontalSpacing(10)
        eq_grid.setVerticalSpacing(4)
        eq_grid.setColumnStretch(1, 1)
        self.eq_sliders = {}
        names = {80: "Low · 80 Hz", 250: "250 Hz", 1000: "1 kHz", 4000: "4 kHz", 12000: "High · 12 kHz"}
        for r, freq in enumerate([80, 250, 1000, 4000, 12000]):
            eq_grid.addWidget(self._form_label(names[freq]), r, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-12, 12)
            slider.setAccessibleName(f"EQ {names[freq]}")
            self.eq_sliders[freq] = slider
            eq_grid.addWidget(slider, r, 1)
            val = QLabel("0 dB")
            val.setObjectName("secondary")
            val.setFixedWidth(44)
            slider.valueChanged.connect(lambda v, lbl=val: lbl.setText(f"{v:+d} dB" if v else "0 dB"))
            eq_grid.addWidget(val, r, 2)
        eq_lay.addLayout(eq_grid)
        layout.addWidget(eq_card)

        fx_card, fx_lay = self._card("Dynamics & Space")
        fx = QGridLayout()
        fx.setHorizontalSpacing(10)
        fx.setVerticalSpacing(8)
        fx.setColumnStretch(2, 1)
        self.compression_check = QCheckBox("Compression")
        fx.addWidget(self.compression_check, 0, 0)
        fx.addWidget(self._form_label("Ratio"), 0, 1)
        self.compression_ratio = QDoubleSpinBox()
        self.compression_ratio.setRange(1.0, 20.0)
        self.compression_ratio.setValue(4.0)
        self.compression_ratio.setSuffix(" : 1")
        fx.addWidget(self.compression_ratio, 0, 2)
        self.reverb_check = QCheckBox("Reverb")
        fx.addWidget(self.reverb_check, 1, 0)
        fx.addWidget(self._form_label("Room"), 1, 1)
        self.reverb_size = QDoubleSpinBox()
        self.reverb_size.setRange(0.0, 1.0)
        self.reverb_size.setSingleStep(0.1)
        self.reverb_size.setValue(0.5)
        fx.addWidget(self.reverb_size, 1, 2)
        self.denoise_check = QCheckBox("De-noise")
        self.denoise_check.setToolTip("Reduce hiss and bleed left over from separation")
        fx.addWidget(self.denoise_check, 2, 0)
        fx.addWidget(self._form_label("Strength"), 2, 1)
        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setRange(0.1, 1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setValue(0.5)
        fx.addWidget(self.denoise_strength, 2, 2)
        fx_lay.addLayout(fx)
        layout.addWidget(fx_card)

        pt_card, pt_lay = self._card("Pitch & Speed", "Practice tools: change key or tempo independently.")
        pt = QHBoxLayout()
        pt.setSpacing(8)
        pt.addWidget(self._form_label("Key"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-12.0, 12.0)
        self.pitch_spin.setSingleStep(1.0)
        self.pitch_spin.setDecimals(1)
        self.pitch_spin.setSuffix(" st")
        self.pitch_spin.setToolTip("Transpose in semitones without changing speed")
        pt.addWidget(self.pitch_spin)
        pt.addWidget(self._form_label("Speed"))
        self.tempo_spin = QSpinBox()
        self.tempo_spin.setRange(25, 200)
        self.tempo_spin.setSingleStep(5)
        self.tempo_spin.setValue(100)
        self.tempo_spin.setSuffix(" %")
        self.tempo_spin.setToolTip("Slow down or speed up without changing pitch")
        pt.addWidget(self.tempo_spin)
        reset_pt = QPushButton("Reset")
        reset_pt.clicked.connect(lambda: (self.pitch_spin.setValue(0), self.tempo_spin.setValue(100)))
        pt.addWidget(reset_pt)
        pt.addStretch()
        pt_lay.addLayout(pt)
        layout.addWidget(pt_card)

        out_card, out_lay = self._card("Output")
        og = QGridLayout()
        og.setHorizontalSpacing(10)
        og.setVerticalSpacing(8)
        og.setColumnStretch(3, 1)
        og.addWidget(self._form_label("Fade in"), 0, 0)
        self.fade_in_spin = QSpinBox()
        self.fade_in_spin.setRange(0, 10000)
        self.fade_in_spin.setSingleStep(100)
        self.fade_in_spin.setSuffix(" ms")
        og.addWidget(self.fade_in_spin, 0, 1)
        og.addWidget(self._form_label("Fade out"), 0, 2)
        self.fade_out_spin = QSpinBox()
        self.fade_out_spin.setRange(0, 10000)
        self.fade_out_spin.setSingleStep(100)
        self.fade_out_spin.setSuffix(" ms")
        og.addWidget(self.fade_out_spin, 0, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        og.addWidget(self._form_label("Loudness"), 1, 0)
        self.normalize_mode = QComboBox()
        self.normalize_mode.addItems(["Off", "Peak", "Loudness (LUFS)"])
        self.normalize_mode.setToolTip(
            "Peak: loudest sample hits the target (dBFS)\n"
            "Loudness: match perceived loudness with a peak limiter.\n"
            "-14 LUFS Spotify/YouTube · -16 Apple Music · -23 broadcast"
        )
        og.addWidget(self.normalize_mode, 1, 1)
        og.addWidget(self._form_label("Target"), 1, 2)
        self.normalize_target = QDoubleSpinBox()
        self.normalize_target.setRange(-30.0, 0.0)
        self.normalize_target.setSingleStep(0.5)
        self.normalize_target.setValue(-1.0)
        self.normalize_target.setSuffix(" dB")
        self.normalize_target.setEnabled(False)
        self.normalize_mode.currentIndexChanged.connect(self._on_normalize_mode)
        og.addWidget(self.normalize_target, 1, 3, alignment=Qt.AlignmentFlag.AlignLeft)
        out_lay.addLayout(og)
        layout.addWidget(out_card)
        layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(340)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([700, 460])
        return page

    # ── Settings window ─────────────────────────────────────────────────────

    def _build_settings_dialog(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumWidth(520)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        tabs = QTabWidget()
        tabs.addTab(self.create_settings_tab(), "General")
        tabs.addTab(self.create_info_tab(), "Processing Info")
        lay.addWidget(tabs)
        buttons = QHBoxLayout()
        reset_btn = QPushButton("Reset to Defaults…")
        reset_btn.setObjectName("danger_btn")
        reset_btn.clicked.connect(self.reset_settings)
        buttons.addWidget(reset_btn)
        buttons.addStretch()
        done = QPushButton("Done")
        done.setObjectName("primary_btn")
        done.setDefault(True)
        done.clicked.connect(dlg.accept)
        buttons.addWidget(done)
        lay.addLayout(buttons)
        return dlg

    def open_settings(self):
        self.update_info()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def create_settings_tab(self) -> QWidget:
        """General settings: processing device, file format, folder."""
        widget = QWidget()
        form = QGridLayout(widget)
        form.setContentsMargins(16, 16, 16, 16)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.setColumnStretch(1, 1)

        form.addWidget(self._form_label("Processor"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'mps', 'cpu'])
        self.device_combo.setCurrentText(self.config.get('device') or 'cpu')
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        form.addWidget(self.device_combo, 0, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        note = QLabel("cuda: NVIDIA GPU · mps: Apple Silicon · cpu: works everywhere, slowest. "
                      "Changing it reloads the model.")
        note.setObjectName("caption")
        note.setWordWrap(True)
        form.addWidget(note, 1, 1)

        form.addWidget(self._form_label("File format"), 2, 0)
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(8)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3', 'flac', 'ogg'])
        self.format_combo.setCurrentText(self.config.get('output_format', 'wav'))
        self.format_combo.currentTextChanged.connect(lambda v: self._save_setting('output_format', v))
        fmt_row.addWidget(self.format_combo)
        fmt_row.addWidget(QLabel("MP3 bitrate"))
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setRange(128, 320)
        self.bitrate_spin.setSingleStep(32)
        self.bitrate_spin.setSuffix(" kbps")
        self.bitrate_spin.setValue(int(self.config.get('output_bitrate', 320)))
        self.bitrate_spin.valueChanged.connect(lambda v: self._save_setting('output_bitrate', v))
        fmt_row.addWidget(self.bitrate_spin)
        fmt_row.addStretch()
        form.addLayout(fmt_row, 2, 1)
        self.float32_check = QCheckBox("32-bit float WAV (for further editing in a DAW)")
        self.float32_check.setChecked(bool(self.config.get('float32', False)))
        self.float32_check.toggled.connect(lambda v: self._save_setting('float32', v))
        form.addWidget(self.float32_check, 3, 1)

        form.addWidget(self._form_label("Working folder"), 4, 0)
        dir_row = QHBoxLayout()
        dir_row.setSpacing(8)
        self.dir_label = QLabel(self.working_directory)
        self.dir_label.setObjectName("secondary")
        self.dir_label.setWordWrap(True)
        self.dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dir_row.addWidget(self.dir_label, 1)
        dir_btn = QPushButton("Change…")
        dir_btn.clicked.connect(self.change_directory)
        dir_row.addWidget(dir_btn)
        form.addLayout(dir_row, 4, 1)
        form.setRowStretch(5, 1)
        return widget

    def create_info_tab(self) -> QWidget:
        """Processing device and model details."""
        widget = QWidget()
        lay = QVBoxLayout(widget)
        lay.setContentsMargins(16, 16, 16, 16)
        self.info_text = QLabel("Model not loaded yet.")
        self.info_text.setWordWrap(True)
        self.info_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.info_text)
        lay.addStretch()
        return widget

    def browse_audio_file(self):
        """Browse for audio file."""
        start_dir = os.path.dirname(self.recent_files[0]) if self.recent_files else str(Path.home() / "Music")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            start_dir,
            AUDIO_FILTER
        )
        if file_path:
            self.load_audio_file(file_path)

    # ── Batch processing ────────────────────────────────────────────────────

    def start_batch(self, files: Optional[list] = None):
        """Separate many files in a row, saving stems per track."""
        if self._is_busy():
            QMessageBox.information(self, "Busy", "A separation is already running.")
            return
        if self.separator is None:
            QMessageBox.warning(self, "Please wait", "The model is still loading.")
            return
        if not files:
            start_dir = os.path.dirname(self.recent_files[0]) if self.recent_files else str(Path.home() / "Music")
            files, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", start_dir, AUDIO_FILTER)
        if not files:
            return
        out_root = QFileDialog.getExistingDirectory(
            self, f"Save stems for {len(files)} tracks to…", self._ensure_working_dir()
        )
        if not out_root:
            return
        two_stems, stem_sources = self._separation_choice()
        self._apply_separation_params()
        fmt = self._output_format()
        self._batch_worker = BatchWorker(
            self.separator, list(files), out_root, fmt,
            f"{self.bitrate_spin.value()}k", two_stems, stem_sources
        )
        self._batch_worker.progress_update.connect(self.update_progress)
        self._batch_worker.file_done.connect(self._on_batch_file_done)
        self._batch_worker.all_done.connect(self._on_batch_done)
        self._batch_out_root = out_root
        self._set_separating(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Batch: {len(files)} files queued")
        self._batch_worker.start()

    def _on_batch_file_done(self, path: str, ok: bool, message: str):
        name = os.path.basename(path)
        self.statusBar().showMessage(
            f"✓ {name} → {message}" if ok else f"✗ {name}: {message}"
        )

    def _on_batch_done(self, ok: int, failed: int, cancelled: bool):
        self._set_separating(False)
        self.progress_bar.setValue(0 if cancelled else 100)
        summary = f"Batch {'cancelled' if cancelled else 'complete'}: {ok} done, {failed} failed"
        self.progress_label.setText(summary)
        self.statusBar().showMessage(summary)
        if not cancelled and ok:
            self.notify(summary + ".", self._batch_out_root, timeout_ms=30000)

    @staticmethod
    def _open_path(path: str):
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def export_click_track(self):
        """Write a metronome track aligned to the analysed beats."""
        info = self._analysis_result
        if not info:
            return
        base = Path(self.current_audio_path).stem if getattr(self, 'current_audio_path', None) else "track"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Click Track",
            os.path.join(self._ensure_working_dir(), f"{_safe_filename(base)}_click_{info['bpm']:.0f}bpm.wav"),
            EXPORT_FILTERS['wav']
        )
        if not file_path:
            return

        def job():
            from audio_analysis import AudioAnalyzer
            an = AudioAnalyzer()
            click = an.generate_click_track(info['beat_times'], info['duration_sec'])
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            sf.write(file_path, click.T, an.sample_rate)
            return file_path

        self._run_task(job, lambda p: self.notify(f"Click track saved: {os.path.basename(p)}", p))

    def analyze_track(self):
        """Run BPM/key/loudness analysis on the loaded file in a worker thread."""
        if not getattr(self, 'current_audio_path', None):
            QMessageBox.warning(self, "Warning", "Please select an audio file first")
            return
        self.analyze_btn.setEnabled(False)
        self.analysis_label.setText("Analyzing… (tempo, key, loudness)")
        self._analysis_worker = AnalysisWorker(self.current_audio_path)
        self._analysis_worker.finished.connect(self._on_analysis_done)
        self._analysis_worker.error.connect(self._on_analysis_error)
        self._analysis_worker.start()

    def _on_analysis_done(self, result: dict):
        self.analyze_btn.setEnabled(True)
        self._analysis_result = result
        self.click_btn.setEnabled(True)
        lufs = f"  ·  {result['lufs']} LUFS" if result.get('lufs') is not None else ""
        mins, secs = divmod(int(round(result['duration_sec'])), 60)
        self.analysis_label.setText(
            f"{result['bpm']:.0f} BPM  ·  {result['key']} {result['mode']}  ·  "
            f"{mins}:{secs:02d}{lufs}"
        )
        self.analysis_label.setToolTip(
            f"Peak {result['peak_db']} dBFS · RMS {result['rms_db']} dB · "
            f"key confidence {result['key_confidence']:.2f}"
        )
        self.statusBar().showMessage("Analysis complete")

    def _on_analysis_error(self, error: str):
        self.analyze_btn.setEnabled(True)
        self.analysis_label.setText(f"Couldn't analyze this file: {error}")

    def _restyle_label(self, label: QLabel, object_name: str):
        """Swap a label's style class and force Qt to re-apply the stylesheet."""
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)

    def load_audio_file(self, file_path: str):
        """Set the current input file and remember it in recents."""
        self.file_path_label.setText(Path(file_path).stem)
        self.file_path_label.setToolTip(file_path)
        self.source_stack.setCurrentIndex(1)
        self.current_audio_path = file_path
        self._analysis_result = None
        self.click_btn.setEnabled(False)
        self.analysis_label.setToolTip("")
        self.analysis_label.setText(
            f"{Path(file_path).suffix.lstrip('.').upper()} · {os.path.dirname(file_path)}"
        )
        self.show_page(self.PAGE_SEPARATE)
        self._add_recent_file(file_path)
        self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")

    def _add_recent_file(self, file_path: str):
        file_path = os.path.abspath(file_path)
        self.recent_files = [file_path] + [p for p in self.recent_files if p != file_path]
        self.recent_files = self.recent_files[:10]
        self.config.set('recent_files', self.recent_files)
        self.config.save()
        self._refresh_recent_combo()

    def _refresh_recent_combo(self):
        """Refresh the sidebar's recent list and File › Open Recent."""
        self.recent_list.clear()
        for path in self.recent_files:
            item = QListWidgetItem(Path(path).stem)
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.recent_list.addItem(item)
        if not self.recent_files:
            item = QListWidgetItem("No recent files")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent_list.addItem(item)
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        menu = getattr(self, 'recent_menu', None)
        if menu is None:
            return
        menu.clear()
        for path in self.recent_files:
            a = menu.addAction(Path(path).name)
            a.setStatusTip(path)
            a.triggered.connect(lambda checked=False, p=path: self.on_recent_selected(p))
        if self.recent_files:
            menu.addSeparator()
            menu.addAction("Clear Menu").triggered.connect(self._clear_recent)
        else:
            menu.addAction("No recent files").setEnabled(False)

    def _clear_recent(self):
        self.recent_files = []
        self.config.set('recent_files', [])
        self.config.save()
        self._refresh_recent_combo()

    def _on_recent_item(self, item: QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.on_recent_selected(path)

    def on_recent_selected(self, path):
        if not path or not isinstance(path, str):
            return
        if not os.path.exists(path):
            self.recent_files = [p for p in self.recent_files if p != path]
            self.config.set('recent_files', self.recent_files)
            self.config.save()
            self._refresh_recent_combo()
            self.notify(f"“{Path(path).name}” was moved or deleted, so it was removed from Recent.")
            return
        self.load_audio_file(path)

    def on_model_changed(self, model_name: str):
        """Handle model change."""
        current_model = self.model_combo.currentData()
        if current_model is None:
            return
        model_name = current_model
        if self.separator and self.separator.model_name != model_name:
            self._start_model_load(model_name, self.config.get('device'))

    def on_device_changed(self, device: str):
        """Persist the device and reload the current model on it."""
        self._save_setting('device', device)
        if self.separator and self.separator.device != device:
            self._start_model_load(self.separator.model_name, device)

    def load_model(self):
        """Load initial model in background thread."""
        self._start_model_load(
            self.config.get('model', 'htdemucs'), self.config.get('device')
        )

    def _start_model_load(self, model_name: str, device: Optional[str]):
        """Load a model off-thread; controls stay locked until it lands."""
        if self._is_busy():
            QMessageBox.information(
                self, "Busy", "Wait for the running separation to finish before switching."
            )
            if self.separator:
                self.set_model_combo(self.separator.model_name)
            return
        self.statusBar().showMessage(f"Loading {model_name}…")
        self.progress_label.setText(
            f"Loading model {model_name}… (downloaded automatically on first use)"
        )
        self._set_model_controls_enabled(False)
        worker = ModelLoadWorker(model_name, device)
        self._model_workers.append(worker)
        worker.finished.connect(self._on_model_loaded)
        worker.error.connect(self._on_model_load_error)
        worker.finished.connect(lambda _=None, w=worker: self._release_model_worker(w))
        worker.error.connect(lambda _=None, w=worker: self._release_model_worker(w))
        worker.start()

    def _release_model_worker(self, worker):
        # Defer: the signal fires from inside the thread's run(); wait for exit
        def _drop():
            worker.wait()
            if worker in self._model_workers:
                self._model_workers.remove(worker)
            worker.deleteLater()
        QTimer.singleShot(0, _drop)

    def _set_model_controls_enabled(self, enabled: bool):
        self.separate_btn.setEnabled(enabled)
        self.batch_btn.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.device_combo.setEnabled(enabled)

    def _on_model_loaded(self, separator):
        self.separator = separator
        self.config.set('model', separator.model_name)
        self.config.save()
        self.statusBar().showMessage(f"Model ready: {separator.model_name} on {separator.device.upper()}")
        self.progress_label.setText("Ready")
        self.set_model_combo(separator.model_name)
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentText(separator.device)
        self.device_combo.blockSignals(False)
        self.sync_stem_controls()
        self.update_info()
        self._set_model_controls_enabled(True)

    def _on_model_load_error(self, error: str):
        logger.error(f"Model loading error: {error}")
        hint = ""
        low = error.lower()
        if any(k in low for k in ("urlopen", "connection", "getaddrinfo", "timed out", "ssl")):
            hint = "\n\nThe model could not be downloaded — check your internet connection."
        QMessageBox.critical(self, "Error", f"Failed to load model: {error}{hint}")
        self.progress_label.setText("Model failed to load")
        self._set_model_controls_enabled(True)
        if self.separator:
            # Roll the UI back to the model that is actually loaded
            self.model_combo.blockSignals(True)
            self.set_model_combo(self.separator.model_name)
            self.model_combo.blockSignals(False)
            self.sync_stem_controls()
        else:
            self.separate_btn.setEnabled(False)
            self.batch_btn.setEnabled(False)

    def set_model_combo(self, model_name: str):
        """Select a model in the combo box by model id."""
        for index in range(self.model_combo.count()):
            if self.model_combo.itemData(index) == model_name:
                self.model_combo.setCurrentIndex(index)
                return

    def sync_stem_controls(self):
        """Enable only the stem controls supported by the active model."""
        if self.separator:
            model_name = self.separator.model_name
        else:
            model_name = self.model_combo.currentData() or self.config.get('model', 'htdemucs')
        available = set(self.config.MODEL_SOURCES.get(model_name, []))
        if not available:
            available = {'drums', 'bass', 'vocals', 'other'}

        rec = self.config.MODEL_RECOMMENDATIONS.get(model_name, '')
        self.model_info_label.setText(rec)
        self.model_info_label.setVisible(bool(rec))
        self.model_caps_label.setText("Stems: " + ", ".join(sorted(available)))

        for stem, checkbox in self.stem_checkboxes.items():
            enabled = stem in available
            if enabled and not checkbox.isEnabled():
                checkbox.setChecked(True)  # newly offered by this model
            checkbox.setEnabled(enabled)
            checkbox.setVisible(enabled)
            if not enabled:
                checkbox.setChecked(False)

        self.two_stems_combo.blockSignals(True)
        current = self.two_stems_combo.currentData()
        self.two_stems_combo.clear()
        self.two_stems_combo.addItem("All stems", userData=None)
        for stem in [s for s in ALL_STEMS if s in available]:
            self.two_stems_combo.addItem(f"{stem.capitalize()} + everything else", userData=stem)
        idx = self.two_stems_combo.findData(current) if current else 0
        self.two_stems_combo.setCurrentIndex(max(0, idx))
        self.two_stems_combo.blockSignals(False)
        self._on_output_mode_changed()

    def _separation_choice(self):
        """(two_stems, stem_sources) from the current UI selection."""
        selected_stems = [
            stem for stem, checkbox in self.stem_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        ]
        two_stems = self.two_stems_combo.currentData()
        stem_sources = None if two_stems else (selected_stems or None)
        return two_stems, stem_sources

    def _apply_separation_params(self):
        """Push the Advanced segment/shifts values into the loaded separator."""
        seg = self.segment_spin.value()
        self.separator.segment = seg if seg > 0 else None
        self.separator.shifts = self.shifts_spin.value()

    def _set_separating(self, busy: bool):
        self.separate_btn.setEnabled(not busy)
        self.batch_btn.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.cancel_btn.setText("Cancel")

    def cancel_separation(self):
        """Stop the running single or batch separation."""
        running = [w for w in (self.worker, self._batch_worker) if w and w.isRunning()]
        if not running:
            return
        for w in running:
            w.cancel()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancelling…")
        self.progress_label.setText("Cancelling after the current chunk…")

    def start_separation(self):
        """Start audio separation."""
        if self._is_busy():
            return
        if not getattr(self, 'current_audio_path', None):
            QMessageBox.warning(self, "Warning", "Please select an audio file first")
            return

        if not os.path.exists(self.current_audio_path):
            QMessageBox.critical(self, "Error", "Audio file not found")
            return

        if self.separator is None:
            QMessageBox.warning(self, "Please wait", "The model is still loading.")
            return

        self._set_separating(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Separating...")
        self.stem_output_list.clear()
        self.preview_files.clear()
        self.current_preview_stem = None
        self._separated_from = self.current_audio_path

        two_stems, stem_sources = self._separation_choice()
        self._apply_separation_params()
        self._sep_started = time.monotonic()

        self.worker = SeparationWorker(
            self.separator,
            self.current_audio_path,
            {
                **self.config.config,
                "stem_sources": stem_sources,
                "two_stems": two_stems,
            }
        )
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_separation_finished)
        self.worker.error.connect(self.on_separation_error)
        self.worker.cancelled.connect(self.on_separation_cancelled)
        self.worker.start()

    def update_progress(self, msg: str, value: int):
        """Update progress bar."""
        if not self.cancel_btn.isEnabled() and self._is_busy():
            return  # cancelling — keep the "Cancelling…" message
        self.progress_label.setText(msg)
        self.progress_bar.setValue(value)

    def on_separation_finished(self, stems: Dict[str, np.ndarray]):
        """Handle separation completion."""
        self.stems = stems
        took = max(0.0, time.monotonic() - getattr(self, '_sep_started', time.monotonic()))
        self.progress_label.setText(f"Done: {len(stems)} stems in {took:.0f} s")
        self.progress_bar.setValue(100)
        self._set_separating(False)
        self.statusBar().showMessage("Separation complete")
        self.populate_stem_outputs()
        self.rebuild_mixer(list(stems.keys()))
        track = Path(getattr(self, '_separated_from', '') or '').stem
        self.mix_subtitle.setText(f"{track} · {len(stems)} stems")
        self.sync_stem_controls()
        if self.auto_save_check.isChecked():
            self._auto_save_stems()

    def _auto_save_stems(self):
        """Write stems to <working dir>/<track>/ without asking."""
        track = Path(getattr(self, '_separated_from', 'track')).stem
        out_dir = os.path.join(self._ensure_working_dir(), _safe_filename(track))
        fmt = self._output_format()
        bitrate = f"{self.bitrate_spin.value()}k"
        float32 = fmt == 'wav' and self.float32_check.isChecked()
        stems = dict(self.stems)
        editor = self.editor

        def job():
            os.makedirs(out_dir, exist_ok=True)
            for name, audio in stems.items():
                TrachSplitterGUI.write_export(
                    editor, os.path.join(out_dir, f"{name}.{fmt}"), audio, fmt, bitrate, float32
                )
            return out_dir

        self._run_task(job, lambda d: self.notify(f"Stems saved to {os.path.basename(d)}", d))

    def on_separation_cancelled(self):
        self._set_separating(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Separation cancelled")
        self.statusBar().showMessage("Cancelled")

    def on_separation_error(self, error: str):
        """Handle separation error."""
        self._set_separating(False)
        QMessageBox.critical(self, "Separation Error", self.format_separation_error(error))
        self.progress_label.setText("Error during separation")

    def format_separation_error(self, error: str) -> str:
        """Return a more actionable separation error message."""
        error_lower = error.lower()

        if "ffmpeg" in error_lower or "could not read the file" in error_lower:
            return (
                "Demucs could not read this audio file.\n\n"
                "Check that:\n"
                "- the file exists and is not corrupted\n"
                "- FFmpeg is installed and available on PATH\n"
                "- the file format is supported (mp3, wav, flac, ogg, m4a)\n\n"
                f"Details: {error}"
            )

        if "cuda" in error_lower and "out of memory" in error_lower:
            return (
                "CUDA ran out of GPU memory during separation.\n\n"
                "Try:\n"
                "- lowering the segment length\n"
                "- closing other GPU-heavy apps\n"
                "- switching to CPU in Settings\n\n"
                f"Details: {error}"
            )

        if "no such file" in error_lower or "file not found" in error_lower:
            return f"The selected file could not be found.\n\nDetails: {error}"

        return (
            "Separation failed.\n\n"
            "If this file is unusual or damaged, try converting it to WAV first.\n\n"
            f"Details: {error}"
        )

    def save_stems(self):
        """Save separated stems to disk."""
        if not self.stems:
            QMessageBox.warning(self, "Warning", "No stems to save. Separate an audio file first.")
            return

        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self._ensure_working_dir()
        )

        if not output_dir:
            return
        fmt = self._output_format()
        bitrate = f"{self.bitrate_spin.value()}k"
        float32 = fmt == 'wav' and self.float32_check.isChecked()
        stems = dict(self.stems)
        editor = self.editor
        self._set_rendering(True, f"Saving {len(stems)} stems…")

        def job():
            os.makedirs(output_dir, exist_ok=True)
            for stem_name, audio in stems.items():
                output_path = os.path.join(output_dir, f"{stem_name}.{fmt}")
                TrachSplitterGUI.write_export(editor, output_path, audio, fmt, bitrate, float32)
            return output_dir

        def on_done(where):
            self._set_rendering(False)
            self.notify(f"Saved {len(stems)} stems to {os.path.basename(where) or where}", where)

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Failed to save stems: {msg}")

        self._run_task(job, on_done, on_error)

    def write_stem_file(self, output_path: str, audio: np.ndarray):
        """Write one stem to disk with a stable channel layout."""
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        if audio.shape[0] not in (1, 2):
            audio = audio[:2]
        sf.write(output_path, audio.T, 44100)

    def populate_stem_outputs(self):
        """Populate the stem list after separation."""
        self.stem_output_list.clear()
        self.preview_files.clear()
        self.stop_playback()
        self.player.setSource(QUrl())  # release file handle before deleting temp dir

        if self.preview_temp_dir and os.path.isdir(self.preview_temp_dir):
            shutil.rmtree(self.preview_temp_dir, ignore_errors=True)

        self.preview_temp_dir = tempfile.mkdtemp(prefix="trachsplitter_stems_")

        # List appears immediately; preview WAVs are written off-thread
        order = {s: i for i, s in enumerate(ALL_STEMS)}
        for stem_name in sorted(self.stems, key=lambda s: (order.get(s.replace('no_', ''), 99), s)):
            item = QListWidgetItem(
                ui_theme.dot_icon(ui_theme.stem_color(stem_name, self.dark)),
                ui_theme.stem_label(stem_name),
            )
            item.setData(Qt.ItemDataRole.UserRole, stem_name)
            self.stem_output_list.addItem(item)
        self.stems_empty_label.setVisible(not self.stems)

        stems_snapshot = dict(self.stems)
        temp_dir = self.preview_temp_dir

        def write_previews():
            files = {}
            for stem_name, audio in stems_snapshot.items():
                temp_path = os.path.join(temp_dir, f"{stem_name}.wav")
                self.write_stem_file(temp_path, audio)
                files[stem_name] = temp_path
            return files

        def on_previews_ready(files):
            if self.preview_temp_dir == temp_dir:  # not superseded by a new separation
                self.preview_files = files
                self.statusBar().showMessage("Stem previews ready")

        self._run_task(write_previews, on_previews_ready)

        if self.stem_output_list.count():
            self.stem_output_list.setCurrentRow(0)
            self.show_selected_stem()

    def on_stem_selection_changed(self):
        """Track the selected stem in the list."""
        self.show_selected_stem()

    def _selected_stems(self) -> list:
        """Names of stems currently selected in the output list."""
        names = []
        for item in self.stem_output_list.selectedItems():
            name = item.data(Qt.ItemDataRole.UserRole)
            if name in self.stems:
                names.append(name)
        return names

    @staticmethod
    def _stem_tag(names: list) -> str:
        """Stable identifier for a stem combination."""
        return " + ".join(sorted(names))

    def _mix_selected(self, names: list) -> np.ndarray:
        """Equal-weight mix of the given stems (heavy — call off-thread)."""
        subset = {n: self.stems[n] for n in names}
        return self.editor.mix_stems(subset, normalize=True)

    def show_selected_stem(self):
        """Show the selected stem(s) in the viewer (computed off-thread).

        With a multi-selection the viewer shows the combined mix.
        """
        names = self._selected_stems()
        if not names:
            self.current_preview_stem = None
            self.spectrogram_view.clear_audio()
            return
        tag = self._stem_tag(names)
        self.current_preview_stem = tag
        title = " + ".join(ui_theme.stem_label(n) for n in sorted(names))
        # One stem: its identity colour; a combination: neutral accent
        color = (ui_theme.stem_color(names[0], self.dark) if len(names) == 1
                 else ui_theme.tokens(self.dark)['accent'])
        self.statusBar().showMessage(f"Rendering {title} view…")

        def compute():
            audio = self.stems[names[0]] if len(names) == 1 else self._mix_selected(names)
            return tag, AudioViewer.compute_view_data(audio, 44100)

        def on_ready(result):
            rtag, (spec_data, wave_env, duration) = result
            if rtag != self.current_preview_stem:
                return  # selection changed while computing
            self.spectrogram_view.set_audio_precomputed(
                duration, 44100, title, spec_data, wave_env, color
            )
            self.statusBar().showMessage("Ready")

        self._run_task(compute, on_ready)

    def on_spectrogram_selection(self, start_sec: float, end_sec: float):
        """Handle a dragged selection on the spectrogram."""
        self.statusBar().showMessage(f"Selected {start_sec:.2f}s - {end_sec:.2f}s on {self.current_preview_stem}")

    def open_stem_in_player(self):
        """Open the selected stem in the system's default music player."""
        if not self.current_preview_stem:
            QMessageBox.warning(self, "Warning", "Select a separated stem first.")
            return
        file_path = self.preview_files.get(self.current_preview_stem)
        if not file_path or not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "Preview file is missing.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", file_path])
            else:
                subprocess.Popen(["xdg-open", file_path])
            self.statusBar().showMessage(f"Opened {self.current_preview_stem} in system player")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open player: {e}")

    # ── In-app playback ─────────────────────────────────────────────────────

    def _play_file(self, path: str, tag: str):
        """Load a file into the in-app player and start playback."""
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(path))
        self._playing_tag = tag
        self.player.play()

    def play_selected_stem(self):
        """Play the selected stem(s) inside the app — several play as one mix."""
        names = self._selected_stems()
        if not names:
            QMessageBox.warning(self, "Warning", "Select a separated stem first.")
            return
        if len(names) == 1:
            path = self.preview_files.get(names[0])
            if not path or not os.path.exists(path):
                QMessageBox.information(
                    self, "Preparing",
                    "Stem preview is still being written — try again in a moment."
                )
                return
            self._play_file(path, self._stem_tag(names))
            self.statusBar().showMessage(f"Playing {names[0]}")
            return

        # Multi-selection: mix off-thread, then play the combined file
        tag = self._stem_tag(names)
        if self._mix_temp_dir is None:
            self._mix_temp_dir = tempfile.mkdtemp(prefix="trachsplitter_mix_")
        out_path = os.path.join(self._mix_temp_dir, f"combo_{int(time.time() * 1000)}.wav")
        self.statusBar().showMessage(f"Mixing {tag}…")

        def job():
            mix = self._mix_selected(names)
            sf.write(out_path, mix.T, 44100)
            return out_path

        def on_done(path):
            self._play_file(path, tag)
            self.statusBar().showMessage(f"Playing {tag}")

        self._run_task(job, on_done)

    def stop_playback(self):
        """Stop in-app playback."""
        self.player.stop()
        self._playing_tag = None
        self.spectrogram_view.update_playhead(None)

    def seek_playback(self, seconds: float):
        """Seek within (or start) playback of the stem shown in the viewer."""
        if not self.current_preview_stem:
            return
        if self._playing_tag != self.current_preview_stem:
            # Single stems can start playing from a click; combined mixes
            # need rendering first (use Play Stem for those)
            path = self.preview_files.get(self.current_preview_stem)
            if not path or not os.path.exists(path):
                return
            self._play_file(path, self.current_preview_stem)
        self.player.setPosition(int(seconds * 1000))

    def _on_player_position(self, position_ms: int):
        if self._playing_tag and self._playing_tag == self.current_preview_stem:
            self.spectrogram_view.update_playhead(position_ms / 1000.0)

    def stop_preview(self):
        """Stop button on the editing tab."""
        self.stop_playback()

    def _get_mix_weights(self) -> Dict[str, float]:
        """Return per-stem linear weights respecting mute/solo state."""
        any_solo = any(btn.isChecked() for btn in self.stem_solos.values())
        weights = {}
        for stem, slider in self.stem_sliders.items():
            if stem not in self.stems:
                continue
            muted = self.stem_mutes.get(stem) and self.stem_mutes[stem].isChecked()
            soloed = self.stem_solos.get(stem) and self.stem_solos[stem].isChecked()
            if muted:
                continue
            if any_solo and not soloed:
                continue
            weights[stem] = 10 ** (slider.value() / 20.0)
        return weights

    def rebuild_mixer(self, stem_names: list):
        """Create one mixer row per separated stem (4, 6 or karaoke pair)."""
        while self.mixer_rows_layout.count():
            item = self.mixer_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.mixer_rows.clear()
        self.stem_sliders.clear()
        self.stem_labels.clear()
        self.stem_mutes.clear()
        self.stem_solos.clear()
        self.stem_pans.clear()

        self.mixer_dots = {}
        order = {s: i for i, s in enumerate(ALL_STEMS)}
        names = sorted(stem_names, key=lambda s: (order.get(s.replace('no_', ''), 99), s))
        for stem in names:
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(8)

            label = ui_theme.stem_label(stem)
            dot = QLabel()
            dot.setFixedWidth(12)
            dot.setPixmap(ui_theme.dot_icon(ui_theme.stem_color(stem, self.dark)).pixmap(12, 12))
            self.mixer_dots[stem] = dot
            row.addWidget(dot)
            name_lbl = QLabel(label)
            name_lbl.setMinimumWidth(60)
            row.addWidget(name_lbl, 2)

            mute_btn = QPushButton("M")
            mute_btn.setObjectName("mute_btn")
            mute_btn.setCheckable(True)
            mute_btn.setFixedSize(28, 28)
            mute_btn.setToolTip(f"Mute {label}")
            mute_btn.setAccessibleName(f"Mute {label}")
            self.stem_mutes[stem] = mute_btn
            row.addWidget(mute_btn)

            solo_btn = QPushButton("S")
            solo_btn.setObjectName("solo_btn")
            solo_btn.setCheckable(True)
            solo_btn.setFixedSize(28, 28)
            solo_btn.setToolTip(f"Solo {label}: silence every other stem")
            solo_btn.setAccessibleName(f"Solo {label}")
            self.stem_solos[stem] = solo_btn
            solo_btn.clicked.connect(lambda checked, s=stem: self._handle_solo(s, checked))
            row.addWidget(solo_btn)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-60)
            slider.setMaximum(12)
            slider.setValue(0)
            slider.setToolTip("Volume in dB")
            slider.setAccessibleName(f"{label} volume")
            self.stem_sliders[stem] = slider
            row.addWidget(slider, 5)

            db_lbl = QLabel("0 dB")
            db_lbl.setObjectName("secondary")
            db_lbl.setFixedWidth(52)
            db_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.stem_labels[stem] = db_lbl
            slider.valueChanged.connect(
                lambda val, s=stem, lbl=db_lbl: self.update_slider_label(s, val, lbl)
            )
            row.addWidget(db_lbl)

            pan_slider = QSlider(Qt.Orientation.Horizontal)
            pan_slider.setObjectName("pan")
            pan_slider.setMinimum(-100)
            pan_slider.setMaximum(100)
            pan_slider.setValue(0)
            pan_slider.setToolTip("Pan: left ↔ right (presets reset to center)")
            pan_slider.setAccessibleName(f"{label} pan")
            self.stem_pans[stem] = pan_slider
            row.addWidget(pan_slider, 2)

            self.mixer_rows[stem] = row_w
            self.mixer_rows_layout.addWidget(row_w)
        self.mixer_empty_label.setVisible(not names)

    def apply_preset(self, preset: str):
        """Apply a named preset: reset all controls, then mute per preset."""
        active = preset_active_stems(preset, self.stem_sliders.keys())
        for stem in self.stem_sliders:
            self.stem_sliders[stem].setValue(0)
            if stem in self.stem_mutes:
                self.stem_mutes[stem].setChecked(stem not in active)
            if stem in self.stem_solos:
                self.stem_solos[stem].setChecked(False)
            if stem in self.stem_pans:
                self.stem_pans[stem].setValue(0)
        if self.stem_sliders:
            self.statusBar().showMessage(f"Preset: {preset}")

    def _handle_solo(self, stem: str, checked: bool):
        """When a stem is soloed, un-solo all others."""
        if checked:
            for s, btn in self.stem_solos.items():
                if s != stem:
                    btn.setChecked(False)

    def quick_export_mix(self):
        """Export a preset mix directly from the Separation tab."""
        if not self.stems:
            QMessageBox.warning(self, "Warning", "No stems — separate audio first.")
            return
        preset = self.quick_preset_combo.currentText()
        weights = {s: 1.0 for s in preset_active_stems(preset, self.stems.keys())}
        if not weights:
            QMessageBox.warning(self, "Warning", "No active stems for this preset.")
            return
        safe_name = preset.lower().replace(" / ", "_").replace(" ", "_").replace("+", "and")
        track = Path(getattr(self, '_separated_from', '') or 'mix').stem
        fmt = self._output_format()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As",
            os.path.join(self._ensure_working_dir(), f"{_safe_filename(track)}_{safe_name}.{fmt}"),
            ";;".join(EXPORT_FILTERS.values()), EXPORT_FILTERS.get(fmt)
        )
        if not file_path:
            return
        active = {k: v for k, v in self.stems.items() if k in weights}
        editor = self.editor
        efmt, bitrate, float32 = self._export_params(file_path)
        self._set_rendering(True, "Exporting mix…")

        def job():
            mix = editor.mix_stems(active, weights, normalize=True)
            return TrachSplitterGUI.write_export(editor, file_path, mix, efmt, bitrate, float32)

        def on_done(written):
            self._set_rendering(False)
            self.statusBar().showMessage(f"Saved: {os.path.basename(written)}")
            self.notify(f"Mix saved: {os.path.basename(written)}", written)

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Export failed: {msg}")

        self._run_task(job, on_done, on_error)

    def export_merged_selection(self):
        """Merge the stems selected in the output list into one audio file."""
        names = self._selected_stems()
        if not names:
            QMessageBox.warning(
                self, "Warning",
                "Select one or more stems in the list first (Ctrl/Shift-click for several)."
            )
            return
        fmt = self._output_format()
        default_name = "_".join(sorted(names)) + f"_merged.{fmt}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Merged Stems As",
            os.path.join(self._ensure_working_dir(), default_name),
            ";;".join(EXPORT_FILTERS.values()), EXPORT_FILTERS.get(fmt)
        )
        if not file_path:
            return
        editor = self.editor
        efmt, bitrate, float32 = self._export_params(file_path)
        self._set_rendering(True, f"Merging {len(names)} stems…")

        def job():
            mix = self._mix_selected(names)
            return TrachSplitterGUI.write_export(editor, file_path, mix, efmt, bitrate, float32)

        def on_done(written):
            self._set_rendering(False)
            self.statusBar().showMessage(f"Merged mix saved: {os.path.basename(written)}")
            self.notify(f"Merged {len(names)} stems: {os.path.basename(written)}", written)

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Merge failed: {msg}")

        self._run_task(job, on_done, on_error)

    def update_slider_label(self, stem: str, value: int, label: QLabel):
        """Update slider value label."""
        label.setText(f"{value:+d} dB" if value else "0 dB")

    def _collect_render_params(self) -> Dict:
        """Snapshot all mixer/effect widget state (UI thread only)."""
        return {
            'weights': self._get_mix_weights(),
            'pans': {s: sl.value() / 100.0 for s, sl in self.stem_pans.items()},
            'eq_freqs': list(self.eq_sliders.keys()),
            'eq_gains': [float(s.value()) for s in self.eq_sliders.values()],
            'denoise': self.denoise_check.isChecked(),
            'denoise_strength': self.denoise_strength.value(),
            'compress': self.compression_check.isChecked(),
            'comp_ratio': self.compression_ratio.value(),
            'reverb': self.reverb_check.isChecked(),
            'reverb_room': self.reverb_size.value(),
            'fade_in_ms': self.fade_in_spin.value(),
            'fade_out_ms': self.fade_out_spin.value(),
            'normalize_mode': self.normalize_mode.currentText(),
            'normalize_db': self.normalize_target.value(),
            'pitch': self.pitch_spin.value(),
            'tempo': self.tempo_spin.value() / 100.0,
        }

    def _on_normalize_mode(self, index: int):
        """Swap the target range/units to match the normalize mode."""
        self.normalize_target.setEnabled(index != 0)
        self.normalize_target.blockSignals(True)
        if index == 2:
            self.normalize_target.setSuffix(" LUFS")
            self.normalize_target.setValue(float(self.config.get('loudness_target', -14.0)))
        else:
            self.normalize_target.setSuffix(" dB")
            self.normalize_target.setValue(-1.0)
        self.normalize_target.blockSignals(False)

    @staticmethod
    def render_mix(stems: Dict[str, np.ndarray], params: Dict, editor: AudioEditor) -> Optional[np.ndarray]:
        """Pure DSP render — no widget access, safe on a worker thread."""
        weights = params['weights']
        if not weights:
            return None
        active = {}
        for name, audio in stems.items():
            if name not in weights:
                continue
            pan = params['pans'].get(name, 0.0)
            active[name] = editor.apply_pan(audio, pan) if abs(pan) > 0.005 else audio
        mix = editor.mix_stems(active, weights, normalize=True)

        if any(abs(g) >= 0.1 for g in params['eq_gains']):
            mix = editor.apply_eq(mix, params['eq_freqs'], params['eq_gains'], shelf_edges=True)
        if params['denoise']:
            mix = editor.reduce_noise(mix, strength=params['denoise_strength'])
        if params['compress']:
            mix = editor.apply_compression(mix, ratio=params['comp_ratio'])
        if params['reverb']:
            mix = editor.apply_reverb(mix, room_size=params['reverb_room'])
        if abs(params.get('pitch', 0.0)) >= 0.05:
            mix = editor.pitch_shift(mix, params['pitch'])
        if abs(params.get('tempo', 1.0) - 1.0) >= 0.005:
            mix = editor.time_stretch(mix, params['tempo'])
        if params['fade_in_ms'] > 0:
            mix = editor.fade_in(mix, params['fade_in_ms'])
        if params['fade_out_ms'] > 0:
            mix = editor.fade_out(mix, params['fade_out_ms'])
        mode = params.get('normalize_mode', 'Off')
        if mode == 'Peak':
            mix = editor.normalize(mix, params['normalize_db'])
        elif mode.startswith('Loudness'):
            mix = editor.normalize_loudness(mix, target_lufs=params['normalize_db'])
        else:
            # Effects (EQ boosts, reverb, pitch) can push past full scale
            mix = editor.limit(mix, -0.3)
        return mix

    def _render_mix(self) -> Optional[np.ndarray]:
        """Synchronous render (blocks — production paths use workers)."""
        return self.render_mix(dict(self.stems), self._collect_render_params(), self.editor)

    def _output_format(self) -> str:
        return self.format_combo.currentText() if hasattr(self, 'format_combo') else 'wav'

    def _export_params(self, file_path: str):
        """Snapshot export format settings (UI thread only)."""
        ext = Path(file_path).suffix.lower().lstrip('.')
        fmt = ext if ext in ('wav', 'mp3', 'flac', 'ogg') else self._output_format()
        bitrate = f"{self.bitrate_spin.value()}k" if hasattr(self, 'bitrate_spin') else '320k'
        float32 = fmt == 'wav' and hasattr(self, 'float32_check') and self.float32_check.isChecked()
        return fmt, bitrate, float32

    @staticmethod
    def write_export(editor: AudioEditor, file_path: str, audio: np.ndarray,
                     fmt: str, bitrate: str, float32: bool) -> str:
        """Write audio to disk — no widget access, safe on a worker thread."""
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        if audio.shape[0] not in (1, 2):
            audio = audio[:2]
        if float32:
            sf.write(file_path, audio.T, 44100, subtype='FLOAT')
            return file_path
        return editor.export_audio(audio, file_path, audio_format=fmt, bitrate=bitrate)

    def _export_audio_file(self, file_path: str, audio: np.ndarray) -> str:
        """Synchronous export honoring settings (blocks — prefer workers)."""
        fmt, bitrate, float32 = self._export_params(file_path)
        return self.write_export(self.editor, file_path, audio, fmt, bitrate, float32)

    def preview_mix(self):
        """Render current mix off-thread, then play it in-app."""
        if not self.stems:
            QMessageBox.warning(self, "Warning", "No stems — separate audio first.")
            return
        params = self._collect_render_params()
        if not params['weights']:
            QMessageBox.warning(self, "Warning", "All stems are muted.")
            return
        if self._mix_temp_dir is None:
            self._mix_temp_dir = tempfile.mkdtemp(prefix="trachsplitter_mix_")
        # Unique name: Windows locks the file while QMediaPlayer plays it
        mix_path = os.path.join(self._mix_temp_dir, f"preview_{int(time.time() * 1000)}.wav")
        stems = dict(self.stems)
        editor = self.editor
        self._set_rendering(True, "Rendering mix preview…")

        def job():
            mix = TrachSplitterGUI.render_mix(stems, params, editor)
            sf.write(mix_path, mix.T, 44100)
            return mix_path

        def on_done(path):
            self._set_rendering(False)
            self._play_file(path, "__mix__")
            self.statusBar().showMessage("Playing mix preview")

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Preview failed: {msg}")

        self._run_task(job, on_done, on_error)

    def export_mix(self):
        """Export the mixed audio."""
        if not self.stems:
            QMessageBox.warning(self, "Warning", "No stems available. Separate an audio file first.")
            return

        fmt = self._output_format()
        track = Path(getattr(self, '_separated_from', '') or 'mix').stem
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mix As",
            os.path.join(self._ensure_working_dir(), f"{_safe_filename(track)}_mix.{fmt}"),
            ";;".join(EXPORT_FILTERS.values()),
            EXPORT_FILTERS.get(fmt, EXPORT_FILTERS['wav'])
        )

        if not file_path:
            return
        params = self._collect_render_params()
        if not params['weights']:
            QMessageBox.warning(self, "Warning", "All stems are muted.")
            return
        stems = dict(self.stems)
        editor = self.editor
        efmt, bitrate, float32 = self._export_params(file_path)
        self._set_rendering(True, "Exporting mix…")

        def job():
            mix = TrachSplitterGUI.render_mix(stems, params, editor)
            return TrachSplitterGUI.write_export(editor, file_path, mix, efmt, bitrate, float32)

        def on_done(written):
            self._set_rendering(False)
            self.statusBar().showMessage(f"Mix saved: {os.path.basename(written)}")
            self.notify(f"Mix saved: {os.path.basename(written)}", written)

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Export failed: {msg}")

        self._run_task(job, on_done, on_error)

    def export_selection(self):
        """Export the drag-selected time region of the selected stem(s).

        Multiple selected stems are merged into one file.
        """
        names = self._selected_stems()
        if not names:
            QMessageBox.warning(self, "Warning", "Select a separated stem first.")
            return
        selection = self.spectrogram_view.selection
        if not selection:
            QMessageBox.warning(
                self, "Warning",
                "Drag-select a region on the spectrogram first (click and drag on the lower plot)."
            )
            return
        length = self.stems[names[0]].shape[-1]
        start = max(0, int(selection[0] * 44100))
        end = min(length, int(selection[1] * 44100))
        if end - start < 441:  # <10 ms
            QMessageBox.warning(self, "Warning", "Selection is too short.")
            return
        fmt = self._output_format()
        base = "_".join(sorted(names))
        default_name = f"{base}_{selection[0]:.1f}s-{selection[1]:.1f}s.{fmt}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Selection As",
            os.path.join(self._ensure_working_dir(), default_name),
            ";;".join(EXPORT_FILTERS.values()), EXPORT_FILTERS.get(fmt)
        )
        if not file_path:
            return
        editor = self.editor
        efmt, bitrate, float32 = self._export_params(file_path)
        self._set_rendering(True, "Exporting selection…")

        def job():
            # Slice before mixing: only the selected region is processed
            clips = {n: self.stems[n][..., start:end] for n in names}
            clip = clips[names[0]] if len(names) == 1 else editor.mix_stems(clips, normalize=True)
            return TrachSplitterGUI.write_export(editor, file_path, clip, efmt, bitrate, float32)

        def on_done(written):
            self._set_rendering(False)
            self.statusBar().showMessage(f"Selection saved: {os.path.basename(written)}")
            self.notify(f"Selection saved: {os.path.basename(written)}", written)

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Export failed: {msg}")

        self._run_task(job, on_done, on_error)

    def change_directory(self):
        """Change working directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            self._ensure_working_dir()
        )
        if dir_path:
            self.working_directory = dir_path
            self.dir_label.setText(dir_path)
            self._save_setting('working_directory', dir_path)

    def _ensure_working_dir(self) -> str:
        """Working folder that is guaranteed to exist.

        Windows' Save dialog shows "File not found" when the suggested path
        points into a folder that doesn't exist (the default
        Music/TrachSplitter is never created otherwise).
        """
        for candidate in (self.working_directory,
                          str(Path.home() / "Music" / "TrachSplitter"),
                          str(Path.home())):
            try:
                os.makedirs(candidate, exist_ok=True)
                if candidate != self.working_directory:
                    self.working_directory = candidate
                    if hasattr(self, 'dir_label'):
                        self.dir_label.setText(candidate)
                return candidate
            except OSError:
                continue
        return str(Path.home())

    def _save_setting(self, key: str, value):
        """Persist a single setting immediately."""
        self.config.set(key, value)
        self.config.save()

    def reset_settings(self):
        """Reset to default settings."""
        reply = QMessageBox.question(
            self,
            "Confirm",
            "Reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.config.reset_to_defaults()
            for combo, value in (
                (self.device_combo, self.config.get('device')),
                (self.format_combo, self.config.get('output_format')),
            ):
                combo.blockSignals(True)
                combo.setCurrentText(value)
                combo.blockSignals(False)
            self.bitrate_spin.setValue(int(self.config.get('output_bitrate', 320)))
            self.float32_check.setChecked(False)
            self.auto_save_check.setChecked(False)
            self.segment_spin.setValue(0)
            self.shifts_spin.setValue(1)
            self.working_directory = str(Path.home() / "Music" / "TrachSplitter")
            self.dir_label.setText(self.working_directory)
            self.load_model()

    def update_info(self):
        """Update device info display."""
        if self.separator:
            info = self.separator.get_device_info()
            available = self.config.MODEL_SOURCES.get(info['model'], [])
            note = self.config.MODEL_NOTES.get(info['model'], 'Unknown capabilities')
            rec = self.config.MODEL_RECOMMENDATIONS.get(info['model'], '')
            group_name = next(
                (group for group, ids in self.config.MODEL_GROUPS.items() if info['model'] in ids),
                'Other'
            )
            rows = [
                ("Processor", info['device'].upper()),
                ("GPU", info['cuda_device']),
                ("Model", f"{info['model']} ({group_name})"),
                ("About this model", note),
                ("Stems", ", ".join(s.capitalize() for s in available)),
            ]
            info_text = "<table cellspacing='6'>" + "".join(
                f"<tr><td style='color:gray'>{k}</td><td>{v}</td></tr>" for k, v in rows
            ) + "</table>"
            self.info_text.setText(info_text)

    def closeEvent(self, event):
        """Save config on exit and clean up temp files."""
        if self._is_busy():
            reply = QMessageBox.question(
                self, "Separation running",
                "A separation is still running. Cancel it and quit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            for w in (self.worker, self._batch_worker):
                if w and w.isRunning():
                    w.cancel()
                    w.wait(30000)
        for w in list(self._model_workers) + list(self._task_workers):
            w.wait(10000)
        self.player.stop()
        self.player.setSource(QUrl())
        geo = self.geometry()
        self.config.set('window_geometry', [geo.x(), geo.y(), geo.width(), geo.height()])
        self.config.save()
        for tmp in [self.preview_temp_dir, self._mix_temp_dir]:
            if tmp and os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
        event.accept()


def main():
    """Run the application."""
    log_dir = Path.home() / '.trachsplitter'
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_dir / 'app.log', encoding='utf-8')]
    if sys.stderr is not None:  # windowed builds have no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("TrachSplitter")
    # Fusion renders stylesheets identically on every OS; colours come from
    # ui_theme and follow the system light/dark appearance
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(ui_theme.ui_font())
    if sys.platform == "win32":
        # Give the process its own taskbar identity so the icon shows
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TrachSplitter.App.1")
    for icon_name in ("icon.ico", "icon.svg"):
        icon_path = Path(__file__).resolve().parent / icon_name
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break
    window = TrachSplitterGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
