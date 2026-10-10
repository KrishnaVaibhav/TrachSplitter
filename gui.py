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
    QProgressBar, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox,
    QListWidget, QListWidgetItem, QMessageBox, QSplitter, QFrame,
    QScrollArea, QSizePolicy, QGridLayout, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer, QUrl
from PySide6.QtGui import QIcon, QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import subprocess
import time

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

from config import Config
from demucs_separator import DemucsSeparator
from audio_editor import AudioEditor
import soundfile as sf
import logging


logger = logging.getLogger(__name__)


class SeparationWorker(QThread):
    """Worker thread for audio separation."""

    progress_update = Signal(str, int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, separator: DemucsSeparator, audio_path: str, config: Dict):
        super().__init__()
        self.separator = separator
        self.audio_path = audio_path
        self.config = config

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
        except Exception as e:
            self.error.emit(str(e))
            logger.error(f"Separation error: {e}\n{traceback.format_exc()}")
        finally:
            self.separator.progress_callback = None


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
    """Widget for displaying a stem spectrogram with click-to-seek."""

    def __init__(self):
        super().__init__()
        self.setObjectName("viewer_frame")
        # Scope to this frame only — a bare QFrame selector would also box
        # every child QLabel (QLabel derives from QFrame)
        self.setStyleSheet(
            "QFrame#viewer_frame { background-color: #1d2125; "
            "border: 1px solid #3a4147; border-radius: 8px; }"
        )
        self.audio = None
        self.sample_rate = 44100
        self.on_seek = None
        self.on_select = None
        self.duration = 0.0
        self.view_start = 0.0
        self.view_end = None
        self.selection = None

        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(8, 4), facecolor="#1a1a1a")
        self.canvas = FigureCanvas(self.figure)
        self.ax_wave = self.figure.add_subplot(211)
        self.ax_spec = self.figure.add_subplot(212, sharex=self.ax_wave)
        for ax in [self.ax_wave, self.ax_spec]:
            ax.set_facecolor("#111111")
            ax.tick_params(colors="#d7f3f1", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#666666")
        self.ax_wave.set_title("Select a stem to inspect", color="#d7f3f1", fontsize=10)
        self.span = None
        self.playhead = None
        self._spec_data = None
        self._wave_env = None
        self._bg = None
        self._ph_artists = None
        self._last_playhead_draw = 0.0
        self.canvas.mpl_connect("resize_event", lambda e: setattr(self, "_bg", None))
        self.canvas.mpl_connect("button_press_event", self._handle_click)
        layout.addWidget(self.canvas)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setObjectName("secondary_btn")
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(0.6))
        toolbar.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setObjectName("secondary_btn")
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(1.6))
        toolbar.addWidget(self.zoom_out_btn)
        self.reset_zoom_btn = QPushButton("Reset")
        self.reset_zoom_btn.setObjectName("secondary_btn")
        self.reset_zoom_btn.clicked.connect(self.reset_zoom)
        toolbar.addWidget(self.reset_zoom_btn)
        toolbar.addSpacing(8)
        self.selection_label = QLabel("Selection: none")
        self.selection_label.setStyleSheet(
            "color: #8fb8b8; font-size: 12px; border: none; background: transparent;"
        )
        toolbar.addWidget(self.selection_label)
        toolbar.addStretch()
        layout.addLayout(toolbar)

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
    ):
        """Display precomputed view data (cheap, UI-thread safe)."""
        self.audio = True  # sentinel: data loaded
        self.sample_rate = sample_rate
        self.duration = duration
        self.view_start = 0.0
        self.view_end = duration
        self._spec_data = spec_data
        self._wave_env = wave_env
        self.selection = None
        self.selection_label.setText("Selection: none")
        self.redraw(title)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 44100, title: str = "Stem Spectrogram"):
        """Synchronous convenience wrapper (blocks — prefer the async path)."""
        spec_data, wave_env, duration = self.compute_view_data(audio, sample_rate)
        self.set_audio_precomputed(duration, sample_rate, title, spec_data, wave_env)

    def clear_audio(self):
        self.audio = None
        self.duration = 0.0
        self.selection = None
        self._spec_data = None
        self._wave_env = None
        self.selection_label.setText("Selection: none")
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
                    ax.axvline(0, color="#00d4ff", linewidth=1.5, animated=True)
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

    def redraw(self, title: str = "Stem Spectrogram", playhead_seconds: Optional[float] = None):
        self._bg = None
        self._ph_artists = None  # cleared with the axes below
        self.ax_wave.clear()
        self.ax_spec.clear()
        if not self.audio:
            self.ax_wave.set_title("Select a stem to inspect", color="#d7f3f1", fontsize=10)
            self.canvas.draw_idle()
            return

        if self._wave_env is not None:
            times, lo, hi = self._wave_env
            self.ax_wave.fill_between(times, lo, hi, color="#7bdff2", linewidth=0)
        self.ax_wave.set_xlim(self.view_start, self.view_end or self.duration)
        self.ax_wave.set_ylabel("Amp", color="#d7f3f1")
        self.ax_wave.set_title(title, color="#d7f3f1", fontsize=10)
        f, t, sxx_db = self._spec_data
        self.ax_spec.pcolormesh(t, f, sxx_db, shading="auto", cmap="magma")
        self.ax_spec.set_ylim(0, min(16000, self.sample_rate // 2))
        self.ax_spec.set_xlim(self.view_start, self.view_end or self.duration)
        self.ax_spec.set_xlabel("Time (s)", color="#d7f3f1")
        self.ax_spec.set_ylabel("Freq (Hz)", color="#d7f3f1")
        self.ax_spec.tick_params(colors="#d7f3f1", labelsize=8)
        if self.span is None:
            self.span = SpanSelector(
                self.ax_spec,
                self._handle_span,
                "horizontal",
                useblit=True,
                props=dict(alpha=0.25, facecolor="#00d4ff")
            )
        if playhead_seconds is not None:
            self.ax_wave.axvline(playhead_seconds, color="#00d4ff", linewidth=1.5)
            self.ax_spec.axvline(playhead_seconds, color="#00d4ff", linewidth=1.5)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _handle_span(self, xmin, xmax):
        self.selection = (float(min(xmin, xmax)), float(max(xmin, xmax)))
        self.selection_label.setText(f"Selection: {self.selection[0]:.2f}s - {self.selection[1]:.2f}s")
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

    def __init__(self, model_name: str, device: str):
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
        self._model_worker: Optional[ModelLoadWorker] = None
        self._mix_temp_dir: Optional[str] = None
        self.stem_mutes: Dict[str, object] = {}
        self.stem_solos: Dict[str, object] = {}
        self.stem_pans: Dict[str, QSlider] = {}
        self.recent_files: list = [
            p for p in self.config.get('recent_files', []) if os.path.exists(p)
        ]

        self._task_workers: list = []
        self._render_buttons: list = []

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._on_player_position)
        self._playing_tag: Optional[str] = None

        self.setAcceptDrops(True)

        self.init_ui()
        self.load_model()

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

    def dragEnterEvent(self, event):
        for url in event.mimeData().urls():
            if Path(url.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS:
                event.acceptProposedAction()
                return

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in AUDIO_EXTENSIONS:
                self.load_audio_file(path)
                event.acceptProposedAction()
                return

    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("TrachSplitter - Audio Source Separation & Editing")
        for icon_name in ("icon.ico", "icon.svg"):
            icon_path = Path(__file__).resolve().parent / icon_name
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                break
        # Responsive geometry: restore last session, else size to the screen
        self.setMinimumSize(1024, 640)
        saved_geo = self.config.get('window_geometry')
        screen = QApplication.primaryScreen().availableGeometry()
        if (isinstance(saved_geo, list) and len(saved_geo) == 4
                and saved_geo[2] >= 1024 and saved_geo[3] >= 640
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
        self.setStyleSheet(self.get_stylesheet())

        # Central widget — 8px grid: 16px page margin, 12px section spacing
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 12, 16, 8)
        main_layout.setSpacing(8)

        # Header: identity left, one-line purpose right (F-pattern scan start)
        header = QHBoxLayout()
        title = QLabel("TrachSplitter")
        title.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        header.addWidget(title)
        subtitle = QLabel("AI stem separation & mixing")
        subtitle.setStyleSheet("color: #8fb8b8; font-size: 13px; padding-top: 6px;")
        header.addWidget(subtitle)
        header.addStretch()
        main_layout.addLayout(header)

        # Create tab widget
        tabs = QTabWidget()

        # Tab 1: Separation
        separation_widget = self.create_separation_tab()
        tabs.addTab(separation_widget, "Separation")

        # Tab 2: Editing
        editing_widget = self.create_editing_tab()
        tabs.addTab(editing_widget, "Editing & Mixing")

        # Tab 3: Settings
        settings_widget = self.create_settings_tab()
        tabs.addTab(settings_widget, "Settings")

        # Tab 4: Info
        info_widget = self.create_info_tab()
        tabs.addTab(info_widget, "Device Info")

        main_layout.addWidget(tabs)

        # Status bar
        self.statusBar().showMessage("Ready")

    def get_stylesheet(self) -> str:
        """Application stylesheet.

        Layout constants follow common HCI guidance: 8px spacing grid,
        >=32px interactive target height (Fitts's law), visible focus and
        hover states, muted secondary text for hierarchy, and a single
        accent hue so the primary action stands out.
        """
        return """
        QMainWindow, QWidget {
            background-color: #23272b;
            color: #e8eef0;
            font-size: 13px;
        }
        QLabel { color: #e8eef0; background: transparent; }
        QLabel#hint { color: #8fb8b8; font-size: 12px; }

        /* ── Buttons: 32px min target, clear hover/pressed/disabled/focus ── */
        QPushButton {
            background-color: #0d7377;
            color: white;
            border: none;
            padding: 6px 14px;
            min-height: 20px;
            border-radius: 6px;
            font-weight: 600;
        }
        QPushButton:hover { background-color: #14919b; }
        QPushButton:pressed { background-color: #0a5463; }
        QPushButton:focus { outline: none; border: 2px solid #7bdff2; padding: 4px 12px; }
        QPushButton:disabled { background-color: #3a4045; color: #7a8288; }

        QPushButton#primary_btn {
            font-size: 15px;
            padding: 10px 20px;
            background-color: #0fa3a8;
        }
        QPushButton#primary_btn:hover { background-color: #17bfc5; }
        QPushButton#primary_btn:pressed { background-color: #0a7276; }

        QPushButton#secondary_btn {
            background-color: #343b41;
            color: #d7e5e5;
        }
        QPushButton#secondary_btn:hover { background-color: #40484f; }

        /* ── Inputs ── */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #2c3237;
            color: #e8eef0;
            border: 1px solid #47505a;
            padding: 5px 8px;
            min-height: 20px;
            border-radius: 5px;
            selection-background-color: #0d7377;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #14919b; }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #7bdff2; }
        QComboBox::drop-down { border: none; width: 24px; }
        QComboBox QAbstractItemView {
            background-color: #2c3237;
            color: #e8eef0;
            selection-background-color: #0d7377;
            border: 1px solid #47505a;
        }
        QCheckBox { spacing: 8px; min-height: 24px; background: transparent; }
        QCheckBox::indicator { width: 18px; height: 18px; }

        /* ── Sliders: large grab handle ── */
        QSlider { min-height: 28px; }
        QSlider::groove:horizontal {
            border: none;
            height: 6px;
            background: #3a4147;
            margin: 2px 0;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal { background: #0d7377; border-radius: 3px; }
        QSlider::handle:horizontal {
            background: #d7f3f1;
            border: 2px solid #0d7377;
            width: 16px;
            height: 16px;
            margin: -7px 0;
            border-radius: 10px;
        }
        QSlider::handle:horizontal:hover { background: #7bdff2; }

        QProgressBar {
            border: none;
            border-radius: 5px;
            background-color: #2c3237;
            color: white;
            min-height: 18px;
            text-align: center;
        }
        QProgressBar::chunk { background-color: #0fa3a8; border-radius: 5px; }

        /* ── Grouping: card-like sections (Gestalt proximity) ── */
        QGroupBox {
            color: #ffffff;
            background-color: #282d32;
            border: 1px solid #3a4147;
            border-radius: 8px;
            margin-top: 14px;
            padding: 20px 12px 12px 12px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 4px;
            padding: 0 6px;
            background-color: #23272b;
            border-radius: 4px;
            color: #9fd8d8;
        }
        QLabel#file_loaded { color: #7bdff2; font-weight: 600; font-size: 13px; }
        QLabel#analysis_result { color: #d7f3f1; font-size: 13px; }

        /* ── Tabs: roomy targets, clear selected state ── */
        QTabWidget::pane { border: 1px solid #3a4147; border-radius: 6px; top: -1px; }
        QTabBar::tab {
            background-color: transparent;
            color: #aeb8be;
            padding: 9px 22px;
            border: none;
            border-bottom: 3px solid transparent;
            font-weight: 600;
        }
        QTabBar::tab:hover { color: #e8eef0; }
        QTabBar::tab:selected { color: #7bdff2; border-bottom: 3px solid #0fa3a8; }

        QListWidget {
            background-color: #2c3237;
            border: 1px solid #3a4147;
            border-radius: 6px;
            padding: 4px;
        }
        QListWidget::item { padding: 6px 8px; border-radius: 4px; min-height: 20px; }
        QListWidget::item:hover { background-color: #343b41; }
        QListWidget::item:selected { background-color: #0d7377; color: white; }

        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical { background: #23272b; width: 10px; border-radius: 5px; }
        QScrollBar::handle:vertical { background: #47505a; border-radius: 5px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #5a656f; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal { background: #23272b; height: 10px; border-radius: 5px; }
        QScrollBar::handle:horizontal { background: #47505a; border-radius: 5px; min-width: 30px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        QSplitter::handle { background-color: #3a4147; }
        QSplitter::handle:horizontal { width: 5px; }
        QSplitter::handle:hover { background-color: #0d7377; }

        QStatusBar { color: #9fb8b8; }
        QToolTip {
            background-color: #16191c;
            color: #d7f3f1;
            border: 1px solid #0d7377;
            padding: 6px 8px;
            border-radius: 4px;
        }

        /* ── Mixer mini-buttons ── */
        QPushButton#mute_btn, QPushButton#solo_btn {
            background-color: #343b41;
            color: #aeb8be;
            font-weight: bold;
            font-size: 11px;
            padding: 0;
            border-radius: 5px;
        }
        QPushButton#mute_btn:checked { background-color: #cc3333; color: white; }
        QPushButton#solo_btn:checked { background-color: #cc8800; color: white; }

        QPushButton#preset_btn { background-color: #2a4a66; padding: 7px 12px; }
        QPushButton#preset_btn:hover { background-color: #2f6a94; }
        """

    def create_separation_tab(self) -> QWidget:
        """Create separation tab.

        Master-detail layout: input workflow reads top-to-bottom in a
        scrollable left column (F-pattern), results and the large
        spectrogram get the remaining space on the right.
        """
        widget = QWidget()
        outer = QHBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # ── LEFT: input workflow ────────────────────────────────────────────
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        # File selection — filename on its own line (never truncates the
        # controls), pickers below
        file_group = QGroupBox("Audio File")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)
        self.file_path_label = QLabel("No file selected — browse, pick a recent file, or drop one anywhere")
        self.file_path_label.setObjectName("hint")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label)
        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        self.recent_combo = QComboBox()
        self.recent_combo.setToolTip("Recently used files")
        self.recent_combo.activated.connect(self.on_recent_selected)
        self._refresh_recent_combo()
        file_row.addWidget(self.recent_combo, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_audio_file)
        file_row.addWidget(browse_btn)
        file_layout.addLayout(file_row)
        layout.addWidget(file_group)

        # Track analysis (BPM / key / loudness)
        analysis_group = QGroupBox("Track Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(8)
        self.analysis_label = QLabel("BPM, key and loudness of the loaded file.")
        self.analysis_label.setObjectName("hint")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        self.analyze_btn = QPushButton("Analyze Track")
        self.analyze_btn.setObjectName("secondary_btn")
        self.analyze_btn.clicked.connect(self.analyze_track)
        analysis_layout.addWidget(self.analyze_btn)
        layout.addWidget(analysis_group)

        # Model selection
        model_group = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout(model_group)

        model_layout.setSpacing(8)
        self.model_combo = QComboBox()
        for group, model_ids in self.config.MODEL_GROUPS.items():
            self.model_combo.addItem(f"—  {group}  —", userData=None)
            idx = self.model_combo.count() - 1
            self.model_combo.model().item(idx).setEnabled(False)
            for model_id in model_ids:
                short = self.config.MODEL_NOTES.get(model_id, model_id)
                self.model_combo.addItem(f"{model_id}  ·  {short}", userData=model_id)
        self.model_combo.setToolTip(
            "Best Quality = slower, cleanest stems\n"
            "Balanced = good speed/quality trade-off\n"
            "Fast / Light = quickest, lowest memory\n"
            "htdemucs_6s adds guitar & piano stems"
        )
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.set_model_combo(self.config.get('model', 'htdemucs'))
        model_layout.addWidget(self.model_combo)

        self.model_info_label = QLabel("")
        self.model_info_label.setObjectName("hint")
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)

        self.model_caps_label = QLabel("")
        self.model_caps_label.setObjectName("hint")
        self.model_caps_label.setWordWrap(True)
        model_layout.addWidget(self.model_caps_label)

        # Advanced options — one labelled row, explained via tooltips
        advanced_h = QHBoxLayout()
        advanced_h.setSpacing(8)
        adv_lbl = QLabel("Advanced:")
        adv_lbl.setObjectName("hint")
        advanced_h.addWidget(adv_lbl)
        advanced_h.addWidget(QLabel("Segment"))
        self.segment_spin = QSpinBox()
        self.segment_spin.setMinimum(0)
        self.segment_spin.setMaximum(60)
        self.segment_spin.setValue(0)
        self.segment_spin.setSpecialValueText("auto")
        self.segment_spin.setSuffix(" s")
        self.segment_spin.setToolTip("Chunk length for processing. auto = let the model decide; lower it if you run out of GPU memory.")
        advanced_h.addWidget(self.segment_spin)
        advanced_h.addWidget(QLabel("Shifts"))
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setMinimum(1)
        self.shifts_spin.setMaximum(10)
        self.shifts_spin.setValue(1)
        self.shifts_spin.setToolTip("Number of prediction passes. More = slightly better quality, proportionally slower.")
        advanced_h.addWidget(self.shifts_spin)
        advanced_h.addStretch()
        model_layout.addLayout(advanced_h)

        layout.addWidget(model_group)

        # Stems selection
        stems_group = QGroupBox("Output Stems")
        stems_layout = QVBoxLayout(stems_group)
        stems_layout.setSpacing(8)
        stems_help = QLabel("Untick any stems you don't need.")
        stems_help.setObjectName("hint")
        stems_help.setWordWrap(True)
        stems_help.setToolTip("Stems the selected model can't produce are hidden automatically.")
        stems_layout.addWidget(stems_help)
        self.stem_checkboxes = {}
        stems_grid = QGridLayout()
        stems_grid.setHorizontalSpacing(16)
        stems_grid.setVerticalSpacing(4)
        for i, stem in enumerate(['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']):
            checkbox = QCheckBox(stem.capitalize())
            checkbox.setChecked(stem in ['drums', 'bass', 'vocals', 'other'])
            self.stem_checkboxes[stem] = checkbox
            stems_grid.addWidget(checkbox, i // 3, i % 3)
        stems_layout.addLayout(stems_grid)
        layout.addWidget(stems_group)
        self.stems_group = stems_group
        self.stems_layout = stems_layout

        # Two stems mode
        two_stems_group = QGroupBox("Karaoke Mode")
        two_stems_layout = QHBoxLayout(two_stems_group)
        two_stems_layout.setSpacing(8)
        extract_lbl = QLabel("Extract only:")
        two_stems_layout.addWidget(extract_lbl)
        self.two_stems_combo = QComboBox()
        self.two_stems_combo.addItem("Disabled")
        self.two_stems_combo.addItems(['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'])
        self.two_stems_combo.setToolTip(
            "Splits the track into just two outputs:\n"
            "the chosen stem, and everything else mixed together.\n"
            "Pick 'vocals' for a karaoke instrumental."
        )
        two_stems_layout.addWidget(self.two_stems_combo, 1)
        layout.addWidget(two_stems_group)
        self.two_stems_group = two_stems_group

        # Primary action + progress: end of the input workflow, most
        # prominent element in the column (visual hierarchy)
        self.separate_btn = QPushButton("Separate Audio")
        self.separate_btn.setObjectName("primary_btn")
        self.separate_btn.clicked.connect(self.start_separation)
        self.separate_btn.setMinimumHeight(46)
        layout.addWidget(self.separate_btn)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)
        self.progress_label = QLabel("Ready")
        self.progress_label.setObjectName("hint")
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(progress_group)

        layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(520)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        # ── RIGHT: results ──────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(10)

        outputs_group = QGroupBox("Separated Outputs")
        outputs_layout = QVBoxLayout(outputs_group)
        self.stem_output_list = QListWidget()
        self.stem_output_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.stem_output_list.setToolTip(
            "Ctrl/Shift-click to select several stems:\n"
            "they play together and merge into one file on export."
        )
        self.stem_output_list.itemSelectionChanged.connect(self.on_stem_selection_changed)
        self.stem_output_list.setMaximumHeight(140)
        self.stem_output_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        outputs_layout.addWidget(self.stem_output_list)

        self.spectrogram_view = AudioViewer()
        self.spectrogram_view.set_on_select(self.on_spectrogram_selection)
        self.spectrogram_view.set_on_seek(self.seek_playback)
        self.spectrogram_view.setMinimumHeight(280)
        self.spectrogram_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        outputs_layout.addWidget(self.spectrogram_view, 1)

        stem_actions = QHBoxLayout()
        stem_actions.setSpacing(8)
        play_stem_btn = QPushButton("▶  Play Stem")
        play_stem_btn.setToolTip("Play the selected stem inside the app. Click the waveform to seek.")
        play_stem_btn.clicked.connect(self.play_selected_stem)
        stem_actions.addWidget(play_stem_btn)

        stop_stem_btn = QPushButton("■  Stop")
        stop_stem_btn.clicked.connect(self.stop_playback)
        stop_stem_btn.setObjectName("secondary_btn")
        stem_actions.addWidget(stop_stem_btn)

        stem_actions.addStretch()

        export_sel_btn = QPushButton("Export Selection…")
        export_sel_btn.setToolTip("Export the region drag-selected on the spectrogram")
        export_sel_btn.clicked.connect(self.export_selection)
        self._render_buttons.append(export_sel_btn)
        stem_actions.addWidget(export_sel_btn)

        open_btn = QPushButton("Open in Music Player")
        open_btn.setObjectName("secondary_btn")
        open_btn.setToolTip("Opens the stem in your system's default audio player")
        open_btn.clicked.connect(self.open_stem_in_player)
        stem_actions.addWidget(open_btn)
        outputs_layout.addLayout(stem_actions)
        right_layout.addWidget(outputs_group, 1)

        # Export shortcuts for the results (grouped with what they act on)
        export_row = QHBoxLayout()
        export_row.setSpacing(8)
        export_row.addWidget(QLabel("Quick Export:"))
        self.quick_preset_combo = QComboBox()
        self.quick_preset_combo.addItems(["Full Mix", "Instrumental", "Vocals Only", "Drums + Bass"])
        self.quick_preset_combo.setToolTip(
            "Full Mix = all stems\n"
            "Instrumental = no vocals\n"
            "Vocals Only = voice track\n"
            "Drums + Bass = rhythm section"
        )
        export_row.addWidget(self.quick_preset_combo, 1)
        quick_export_btn = QPushButton("Export Mix")
        quick_export_btn.clicked.connect(self.quick_export_mix)
        self._render_buttons.append(quick_export_btn)
        export_row.addWidget(quick_export_btn)

        merge_btn = QPushButton("Merge Selected…")
        merge_btn.setToolTip(
            "Merge the stems selected in the list above into a single audio file"
        )
        merge_btn.clicked.connect(self.export_merged_selection)
        self._render_buttons.append(merge_btn)
        export_row.addWidget(merge_btn)

        save_btn = QPushButton("Save Individual Stems")
        save_btn.clicked.connect(self.save_stems)
        self._render_buttons.append(save_btn)
        export_row.addWidget(save_btn)
        right_layout.addLayout(export_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 900])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        # Populate model info / stem availability for the initial selection
        self.sync_stem_controls()
        return widget

    def create_editing_tab(self) -> QWidget:
        """Create editing and mixing tab.

        Two-pane layout: the mixer (used constantly) gets the wide left
        pane; effects and EQ (set-and-forget) sit right. Transport actions
        live in a persistent bottom bar so they never scroll out of reach.
        """
        widget = QWidget()
        outer = QVBoxLayout(widget)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # ── LEFT: mixer + presets ───────────────────────────────────────────
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        # ── Stem Mixer ──────────────────────────────────────────────────────
        mixer_group = QGroupBox("Stem Mixer")
        mixer_layout = QVBoxLayout(mixer_group)

        header = QHBoxLayout()
        for text, stretch in [("Stem", 2), ("M", 0), ("S", 0), ("Volume", 5), ("Level", 1), ("Pan", 2)]:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: #b8d8d8; font-size: 11px;")
            if stretch:
                header.addWidget(lbl, stretch)
            else:
                lbl.setFixedWidth(32)
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                header.addWidget(lbl)
        mixer_layout.addLayout(header)

        self.stem_sliders = {}
        self.stem_labels = {}
        self.stem_mutes = {}
        self.stem_solos = {}

        for stem in ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']:
            row = QHBoxLayout()

            name_lbl = QLabel(f"{stem.capitalize()}")
            name_lbl.setMinimumWidth(60)
            row.addWidget(name_lbl, 2)

            mute_btn = QPushButton("M")
            mute_btn.setObjectName("mute_btn")
            mute_btn.setCheckable(True)
            mute_btn.setFixedSize(28, 28)
            mute_btn.setToolTip("Mute")
            self.stem_mutes[stem] = mute_btn
            row.addWidget(mute_btn)

            solo_btn = QPushButton("S")
            solo_btn.setObjectName("solo_btn")
            solo_btn.setCheckable(True)
            solo_btn.setFixedSize(28, 28)
            solo_btn.setToolTip("Solo — silence all other stems")
            self.stem_solos[stem] = solo_btn
            solo_btn.clicked.connect(lambda checked, s=stem: self._handle_solo(s, checked))
            row.addWidget(solo_btn)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-60)
            slider.setMaximum(12)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(6)
            self.stem_sliders[stem] = slider
            row.addWidget(slider, 5)

            db_lbl = QLabel("  0 dB")
            db_lbl.setMinimumWidth(52)
            self.stem_labels[stem] = db_lbl
            slider.valueChanged.connect(
                lambda val, s=stem, lbl=db_lbl: self.update_slider_label(s, val, lbl)
            )
            row.addWidget(db_lbl, 1)

            pan_slider = QSlider(Qt.Orientation.Horizontal)
            pan_slider.setMinimum(-100)
            pan_slider.setMaximum(100)
            pan_slider.setValue(0)
            pan_slider.setToolTip("Pan: left ↔ right (presets reset to center)")
            pan_slider.setMaximumWidth(120)
            self.stem_pans[stem] = pan_slider
            row.addWidget(pan_slider, 2)

            mixer_layout.addLayout(row)

        layout.addWidget(mixer_group)

        # ── Quick Presets ────────────────────────────────────────────────────
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QGridLayout(presets_group)
        presets_layout.setHorizontalSpacing(8)
        presets_layout.setVerticalSpacing(8)
        preset_defs = [
            ("Full Mix",      {}),
            ("Instrumental",  {"vocals"}),
            ("Vocals Only",   {"drums", "bass", "other", "guitar", "piano"}),
            ("Drums + Bass",  {"vocals", "other", "guitar", "piano"}),
            ("Ambient / Other", {"drums", "bass", "vocals", "guitar", "piano"}),
        ]
        for i, (name, mute_set) in enumerate(preset_defs):
            btn = QPushButton(name)
            btn.setObjectName("preset_btn")
            btn.clicked.connect(lambda checked, ms=mute_set: self.apply_preset(ms))
            presets_layout.addWidget(btn, i // 3, i % 3)
        layout.addWidget(presets_group)
        layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(460)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(left_scroll)

        # ── RIGHT: effects + EQ ─────────────────────────────────────────────
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        # ── Effects ──────────────────────────────────────────────────────────
        effects_group = QGroupBox("Effects")
        effects_layout = QVBoxLayout(effects_group)

        comp_h = QHBoxLayout()
        self.compression_check = QCheckBox("Compression")
        comp_h.addWidget(self.compression_check)
        comp_h.addWidget(QLabel("Ratio:"))
        self.compression_ratio = QDoubleSpinBox()
        self.compression_ratio.setValue(4.0)
        self.compression_ratio.setMinimum(1.0)
        self.compression_ratio.setMaximum(20.0)
        comp_h.addWidget(self.compression_ratio)
        comp_h.addStretch()
        effects_layout.addLayout(comp_h)

        reverb_h = QHBoxLayout()
        self.reverb_check = QCheckBox("Reverb")
        reverb_h.addWidget(self.reverb_check)
        reverb_h.addWidget(QLabel("Room Size:"))
        self.reverb_size = QDoubleSpinBox()
        self.reverb_size.setValue(0.5)
        self.reverb_size.setMinimum(0.0)
        self.reverb_size.setMaximum(1.0)
        self.reverb_size.setSingleStep(0.1)
        reverb_h.addWidget(self.reverb_size)
        reverb_h.addStretch()
        effects_layout.addLayout(reverb_h)

        denoise_h = QHBoxLayout()
        self.denoise_check = QCheckBox("De-noise (spectral gate)")
        self.denoise_check.setToolTip("Reduce hiss/bleed left over from separation")
        denoise_h.addWidget(self.denoise_check)
        denoise_h.addWidget(QLabel("Strength:"))
        self.denoise_strength = QDoubleSpinBox()
        self.denoise_strength.setMinimum(0.1)
        self.denoise_strength.setMaximum(1.0)
        self.denoise_strength.setSingleStep(0.1)
        self.denoise_strength.setValue(0.5)
        denoise_h.addWidget(self.denoise_strength)
        denoise_h.addStretch()
        effects_layout.addLayout(denoise_h)

        fade_h = QHBoxLayout()
        fade_h.addWidget(QLabel("Fade In (ms):"))
        self.fade_in_spin = QSpinBox()
        self.fade_in_spin.setMaximum(10000)
        self.fade_in_spin.setSingleStep(100)
        fade_h.addWidget(self.fade_in_spin)
        fade_h.addWidget(QLabel("Fade Out (ms):"))
        self.fade_out_spin = QSpinBox()
        self.fade_out_spin.setMaximum(10000)
        self.fade_out_spin.setSingleStep(100)
        fade_h.addWidget(self.fade_out_spin)
        fade_h.addStretch()
        effects_layout.addLayout(fade_h)

        norm_h = QHBoxLayout()
        self.normalize_check = QCheckBox("Normalize to")
        norm_h.addWidget(self.normalize_check)
        self.normalize_target = QDoubleSpinBox()
        self.normalize_target.setMinimum(-24.0)
        self.normalize_target.setMaximum(0.0)
        self.normalize_target.setValue(-3.0)
        self.normalize_target.setSuffix(" dB")
        norm_h.addWidget(self.normalize_target)
        norm_h.addStretch()
        effects_layout.addLayout(norm_h)

        layout.addWidget(effects_group)

        # ── 5-Band EQ ────────────────────────────────────────────────────────
        eq_group = QGroupBox("5-Band EQ")
        eq_layout = QVBoxLayout(eq_group)
        self.eq_sliders = {}
        for freq in [80, 250, 1000, 4000, 12000]:
            eq_h = QHBoxLayout()
            lbl = QLabel(f"{freq} Hz:")
            lbl.setMinimumWidth(70)
            eq_h.addWidget(lbl)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-12)
            slider.setMaximum(12)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.eq_sliders[freq] = slider
            eq_h.addWidget(slider)
            eq_val = QLabel("  0 dB")
            eq_val.setMinimumWidth(50)
            slider.valueChanged.connect(lambda v, lbl=eq_val: lbl.setText(f"{v:+d} dB"))
            eq_h.addWidget(eq_val)
            eq_layout.addLayout(eq_h)
        layout.addWidget(eq_group)
        layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(320)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 480])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        # ── Persistent transport bar (never scrolls away) ───────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        play_mix_btn = QPushButton("▶  Preview Mix")
        play_mix_btn.setObjectName("primary_btn")
        play_mix_btn.setMinimumHeight(40)
        play_mix_btn.clicked.connect(self.preview_mix)
        self._render_buttons.append(play_mix_btn)
        action_row.addWidget(play_mix_btn, 2)

        stop_mix_btn = QPushButton("■  Stop")
        stop_mix_btn.setObjectName("secondary_btn")
        stop_mix_btn.setMinimumHeight(40)
        stop_mix_btn.clicked.connect(self.stop_preview)
        action_row.addWidget(stop_mix_btn, 1)

        export_btn = QPushButton("Export Mix…")
        export_btn.setMinimumHeight(40)
        export_btn.clicked.connect(self.export_mix)
        self._render_buttons.append(export_btn)
        action_row.addWidget(export_btn, 2)
        outer.addLayout(action_row)

        return widget

    def create_settings_tab(self) -> QWidget:
        """Create settings tab (single readable column, ~640px measure)."""
        outer_widget = QWidget()
        outer_h = QHBoxLayout(outer_widget)
        widget = QWidget()
        widget.setMaximumWidth(680)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(12)
        outer_h.addWidget(widget)
        outer_h.addStretch()

        # Device
        device_group = QGroupBox("Device")
        device_layout = QVBoxLayout(device_group)

        device_h = QHBoxLayout()
        device_h.addWidget(QLabel("Processing Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cuda', 'cpu'])
        self.device_combo.setCurrentText(self.config.get('device', 'cpu'))
        self.device_combo.currentTextChanged.connect(
            lambda v: self._save_setting('device', v)
        )
        device_h.addWidget(self.device_combo)
        device_layout.addLayout(device_h)
        device_note = QLabel("Device change applies the next time a model is loaded.")
        device_note.setStyleSheet("color: #b8d8d8; font-size: 12px;")
        device_layout.addWidget(device_note)
        layout.addWidget(device_group)

        # Output format
        format_group = QGroupBox("Output Format")
        format_layout = QVBoxLayout(format_group)

        format_h = QHBoxLayout()
        format_h.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['wav', 'mp3', 'flac'])
        self.format_combo.setCurrentText(self.config.get('output_format', 'wav'))
        self.format_combo.currentTextChanged.connect(
            lambda v: self._save_setting('output_format', v)
        )
        format_h.addWidget(self.format_combo)

        format_h.addWidget(QLabel("MP3 Bitrate:"))
        self.bitrate_spin = QSpinBox()
        self.bitrate_spin.setMinimum(128)
        self.bitrate_spin.setMaximum(320)
        self.bitrate_spin.setValue(int(self.config.get('output_bitrate', 320)))
        self.bitrate_spin.valueChanged.connect(
            lambda v: self._save_setting('output_bitrate', v)
        )
        format_h.addWidget(self.bitrate_spin)
        format_layout.addLayout(format_h)

        self.float32_check = QCheckBox("Float32 (WAV only)")
        self.float32_check.setChecked(bool(self.config.get('float32', False)))
        self.float32_check.toggled.connect(
            lambda v: self._save_setting('float32', v)
        )
        format_layout.addWidget(self.float32_check)

        layout.addWidget(format_group)

        # Working directory
        dir_group = QGroupBox("Working Directory")
        dir_layout = QHBoxLayout(dir_group)
        self.dir_label = QLabel(self.working_directory)
        dir_layout.addWidget(self.dir_label)
        dir_btn = QPushButton("Change")
        dir_btn.clicked.connect(self.change_directory)
        dir_layout.addWidget(dir_btn)
        layout.addWidget(dir_group)

        # Reset button — destructive-ish, kept visually secondary
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setObjectName("secondary_btn")
        reset_btn.clicked.connect(self.reset_settings)
        reset_row = QHBoxLayout()
        reset_row.addWidget(reset_btn)
        reset_row.addStretch()
        layout.addLayout(reset_row)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(outer_widget)
        return scroll

    def create_info_tab(self) -> QWidget:
        """Create device info tab."""
        outer_widget = QWidget()
        outer_h = QHBoxLayout(outer_widget)
        widget = QWidget()
        widget.setMaximumWidth(680)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 12, 8, 8)
        outer_h.addWidget(widget)
        outer_h.addStretch()

        info_group = QGroupBox("System Information")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QLabel()
        self.info_text.setWordWrap(True)
        self.info_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)
        layout.addStretch()

        self.update_info()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(outer_widget)
        return scroll

    def browse_audio_file(self):
        """Browse for audio file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            str(Path.home() / "Music"),
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
        )
        if file_path:
            self.load_audio_file(file_path)

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
        self.analysis_label.setText(
            f"<b>{result['bpm']} BPM</b>  ·  "
            f"<b>{result['key']} {result['mode']}</b>  ·  "
            f"{result['duration_sec']} s<br>"
            f"RMS {result['rms_db']} dB  ·  Peak {result['peak_db']} dB  ·  "
            f"key confidence {result['key_confidence']:.2f}"
        )
        self._restyle_label(self.analysis_label, "analysis_result")
        self.statusBar().showMessage("Analysis complete")

    def _on_analysis_error(self, error: str):
        self.analyze_btn.setEnabled(True)
        self._restyle_label(self.analysis_label, "hint")
        self.analysis_label.setText(f"Analysis failed: {error}")

    def _restyle_label(self, label: QLabel, object_name: str):
        """Swap a label's style class and force Qt to re-apply the stylesheet."""
        label.setObjectName(object_name)
        label.style().unpolish(label)
        label.style().polish(label)

    def load_audio_file(self, file_path: str):
        """Set the current input file and remember it in recents."""
        self.file_path_label.setText(f"♪  {os.path.basename(file_path)}")
        self.file_path_label.setToolTip(file_path)
        self._restyle_label(self.file_path_label, "file_loaded")
        self.current_audio_path = file_path
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
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        self.recent_combo.addItem("Recent files…", userData=None)
        for path in self.recent_files:
            self.recent_combo.addItem(os.path.basename(path), userData=path)
            self.recent_combo.setItemData(
                self.recent_combo.count() - 1, path, Qt.ItemDataRole.ToolTipRole
            )
        self.recent_combo.blockSignals(False)

    def on_recent_selected(self, index: int):
        path = self.recent_combo.itemData(index)
        if not path:
            return
        if not os.path.exists(path):
            QMessageBox.warning(self, "Missing", f"File no longer exists:\n{path}")
            self.recent_files = [p for p in self.recent_files if p != path]
            self.config.set('recent_files', self.recent_files)
            self._refresh_recent_combo()
            return
        self.load_audio_file(path)

    def on_model_changed(self, model_name: str):
        """Handle model change."""
        current_model = self.model_combo.currentData()
        if current_model is None:
            return
        model_name = current_model
        if self.separator and self.separator.model_name != model_name:
            self.statusBar().showMessage(f"Loading {model_name}...")
            self.separate_btn.setEnabled(False)
            self._model_worker = ModelLoadWorker(model_name, self.config.get('device', 'cpu'))
            self._model_worker.finished.connect(self._on_model_changed_done)
            self._model_worker.error.connect(self._on_model_load_error)
            self._model_worker.start()

    def _on_model_changed_done(self, separator):
        self.separator = separator
        self.config.set('model', separator.model_name)
        self.statusBar().showMessage(f"Model loaded: {separator.model_name}")
        self.sync_stem_controls()
        self.update_info()
        self.separate_btn.setEnabled(True)

    def load_model(self):
        """Load initial model in background thread."""
        device = self.config.get('device', 'cpu')
        default_model = self.config.get('model', 'htdemucs')
        self.statusBar().showMessage("Loading model...")
        self.separate_btn.setEnabled(False)
        self._model_worker = ModelLoadWorker(default_model, device)
        self._model_worker.finished.connect(self._on_model_loaded)
        self._model_worker.error.connect(self._on_model_load_error)
        self._model_worker.start()

    def _on_model_loaded(self, separator):
        self.separator = separator
        self.config.set('model', separator.model_name)
        self.statusBar().showMessage("Model loaded successfully")
        self.set_model_combo(separator.model_name)
        self.sync_stem_controls()
        self.update_info()
        self.separate_btn.setEnabled(True)

    def _on_model_load_error(self, error: str):
        QMessageBox.critical(self, "Error", f"Failed to load model: {error}")
        logger.error(f"Model loading error: {error}")
        self.separate_btn.setEnabled(True)

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
            checkbox.setEnabled(enabled)
            checkbox.setVisible(enabled)
            if not enabled:
                checkbox.setChecked(False)

        self.two_stems_combo.blockSignals(True)
        current = self.two_stems_combo.currentText()
        self.two_stems_combo.clear()
        self.two_stems_combo.addItem("Disabled")
        for stem in [s for s in ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'] if s in available]:
            self.two_stems_combo.addItem(stem)
        if current in [self.two_stems_combo.itemText(i) for i in range(self.two_stems_combo.count())]:
            self.two_stems_combo.setCurrentText(current)
        else:
            self.two_stems_combo.setCurrentText("Disabled")
        self.two_stems_combo.blockSignals(False)

    def start_separation(self):
        """Start audio separation."""
        if not hasattr(self, 'current_audio_path'):
            QMessageBox.warning(self, "Warning", "Please select an audio file first")
            return

        if not os.path.exists(self.current_audio_path):
            QMessageBox.critical(self, "Error", "Audio file not found")
            return

        self.separate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Separating...")
        self.stem_output_list.clear()
        self.preview_files.clear()
        self.current_preview_stem = None

        selected_stems = [
            stem for stem, checkbox in self.stem_checkboxes.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        ]
        two_stems = self.two_stems_combo.currentText()
        if two_stems == "Disabled":
            two_stems = None

        stem_sources = None if two_stems else (selected_stems or None)

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
        self.worker.start()

    def update_progress(self, msg: str, value: int):
        """Update progress bar."""
        self.progress_label.setText(msg)
        self.progress_bar.setValue(value)

    def on_separation_finished(self, stems: Dict[str, np.ndarray]):
        """Handle separation completion."""
        self.stems = stems
        self.progress_label.setText(f"Completed! Extracted {len(stems)} stems")
        self.progress_bar.setValue(100)
        self.separate_btn.setEnabled(True)
        self.statusBar().showMessage("Separation complete")
        self.populate_stem_outputs()
        self.sync_stem_controls()

    def on_separation_error(self, error: str):
        """Handle separation error."""
        QMessageBox.critical(self, "Separation Error", self.format_separation_error(error))
        self.progress_label.setText("Error during separation")
        self.separate_btn.setEnabled(True)

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
            self.working_directory
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
            QMessageBox.information(self, "Success", f"Stems saved to {where}")

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
        for stem_name in self.stems:
            self.stem_output_list.addItem(stem_name)

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
            name = item.text().split("  (", 1)[0].strip()
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
        title = " + ".join(n.capitalize() for n in sorted(names))
        self.statusBar().showMessage(f"Rendering {title} view…")

        def compute():
            audio = self.stems[names[0]] if len(names) == 1 else self._mix_selected(names)
            return tag, AudioViewer.compute_view_data(audio, 44100)

        def on_ready(result):
            rtag, (spec_data, wave_env, duration) = result
            if rtag != self.current_preview_stem:
                return  # selection changed while computing
            self.spectrogram_view.set_audio_precomputed(
                duration, 44100, f"{title} spectrogram", spec_data, wave_env
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

    def apply_preset(self, mute_set: set):
        """Apply a preset: reset all controls, then mute stems in mute_set."""
        for stem in self.stem_sliders:
            self.stem_sliders[stem].setValue(0)
            if stem in self.stem_mutes:
                self.stem_mutes[stem].setChecked(stem in mute_set)
            if stem in self.stem_solos:
                self.stem_solos[stem].setChecked(False)
            if stem in self.stem_pans:
                self.stem_pans[stem].setValue(0)

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
        preset_mutes = {
            "Full Mix":        set(),
            "Instrumental":    {"vocals"},
            "Vocals Only":     {"drums", "bass", "other", "guitar", "piano"},
            "Drums + Bass":    {"vocals", "other", "guitar", "piano"},
        }
        preset = self.quick_preset_combo.currentText()
        muted = preset_mutes.get(preset, set())
        weights = {s: 1.0 for s in self.stems if s not in muted}
        if not weights:
            QMessageBox.warning(self, "Warning", "No active stems for this preset.")
            return
        safe_name = preset.lower().replace(" ", "_").replace("+", "and")
        fmt = self._output_format()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mix As",
            os.path.join(self.working_directory, f"mix_{safe_name}.{fmt}"),
            "WAV Files (*.wav);;MP3 Files (*.mp3);;FLAC Files (*.flac)"
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
            QMessageBox.information(self, "Saved", f"Mix saved:\n{written}")

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
            os.path.join(self.working_directory, default_name),
            "WAV Files (*.wav);;MP3 Files (*.mp3);;FLAC Files (*.flac)"
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
            QMessageBox.information(self, "Saved", f"Merged {len(names)} stems:\n{written}")

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Merge failed: {msg}")

        self._run_task(job, on_done, on_error)

    def update_slider_label(self, stem: str, value: int, label: QLabel):
        """Update slider value label."""
        label.setText(f"{value:+d} dB")

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
            'normalize': self.normalize_check.isChecked(),
            'normalize_db': self.normalize_target.value(),
        }

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
            mix = editor.apply_eq(mix, params['eq_freqs'], params['eq_gains'])
        if params['denoise']:
            mix = editor.reduce_noise(mix, strength=params['denoise_strength'])
        if params['compress']:
            mix = editor.apply_compression(mix, ratio=params['comp_ratio'])
        if params['reverb']:
            mix = editor.apply_reverb(mix, room_size=params['reverb_room'])
        if params['fade_in_ms'] > 0:
            mix = editor.fade_in(mix, params['fade_in_ms'])
        if params['fade_out_ms'] > 0:
            mix = editor.fade_out(mix, params['fade_out_ms'])
        if params['normalize']:
            mix = editor.normalize(mix, params['normalize_db'])
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
        filters = {
            'wav': "WAV Files (*.wav)",
            'mp3': "MP3 Files (*.mp3)",
            'flac': "FLAC Files (*.flac)",
        }
        all_filters = ";;".join(filters.values())
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mix As",
            os.path.join(self.working_directory, f"mix.{fmt}"),
            all_filters,
            filters.get(fmt, filters['wav'])
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
            QMessageBox.information(self, "Saved", f"Mix saved:\n{written}")

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
        audio = self.stems[names[0]] if len(names) == 1 else self._mix_selected(names)
        start = max(0, int(selection[0] * 44100))
        end = min(audio.shape[-1], int(selection[1] * 44100))
        if end - start < 441:  # <10 ms
            QMessageBox.warning(self, "Warning", "Selection is too short.")
            return
        fmt = self._output_format()
        base = "_".join(sorted(names))
        default_name = f"{base}_{selection[0]:.1f}s-{selection[1]:.1f}s.{fmt}"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Selection As",
            os.path.join(self.working_directory, default_name),
            "WAV Files (*.wav);;MP3 Files (*.mp3);;FLAC Files (*.flac)"
        )
        if not file_path:
            return
        clip = audio[..., start:end]
        editor = self.editor
        efmt, bitrate, float32 = self._export_params(file_path)
        self._set_rendering(True, "Exporting selection…")

        def job():
            return TrachSplitterGUI.write_export(editor, file_path, clip, efmt, bitrate, float32)

        def on_done(written):
            self._set_rendering(False)
            self.statusBar().showMessage(f"Selection saved: {os.path.basename(written)}")
            QMessageBox.information(self, "Saved", f"Selection saved:\n{written}")

        def on_error(msg):
            self._set_rendering(False)
            QMessageBox.critical(self, "Error", f"Export failed: {msg}")

        self._run_task(job, on_done, on_error)

    def change_directory(self):
        """Change working directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Working Directory",
            self.working_directory
        )
        if dir_path:
            self.working_directory = dir_path
            self.dir_label.setText(dir_path)
            self._save_setting('working_directory', dir_path)

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
            info_text = f"""
            <b>Device:</b> {info['device'].upper()}<br>
            <b>CUDA Available:</b> {info['cuda_available']}<br>
            <b>GPU:</b> {info['cuda_device']}<br>
            <b>Model:</b> {info['model']}<br>
            <b>Tier:</b> {group_name}<br>
            <b>Model Notes:</b> {note}<br>
            <b>Recommendation:</b> {rec}<br>
            <b>Supported Stems:</b> {', '.join(available)}<br>
            <b>Sources:</b> {info['sources']}<br>
            """
            self.info_text.setText(info_text)

    def closeEvent(self, event):
        """Save config on exit and clean up temp files."""
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
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
