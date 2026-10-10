# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for TrachSplitter.
# Use via:  python build.py
# Or directly:  pyinstaller TrachSplitter.spec
#
# Cross-platform: this spec auto-detects the OS and adjusts accordingly.

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPEC_PATH).parent  # noqa: F821  (SPEC_PATH injected by PyInstaller)
IS_WIN   = sys.platform == "win32"
IS_MAC   = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

APP_NAME = "TrachSplitter"
VERSION  = "1.0.0"

# ── Source data files ──────────────────────────────────────────────────────────

datas = []

# Local project source files
for name in ["config.py", "demucs_separator.py", "audio_editor.py"]:
    src = ROOT / name
    if src.exists():
        datas.append((str(src), "."))

# Matplotlib data (style sheets, fonts)
datas += collect_data_files("matplotlib")

# Demucs package data (YAML configs, etc.)
datas += collect_data_files("demucs")

# Librosa data (impulse response files etc.)
datas += collect_data_files("librosa")

# Soundfile — libsndfile shared library is auto-detected via binaries below


# ── Hidden imports ─────────────────────────────────────────────────────────────

hiddenimports = [
    # PyTorch core
    *collect_submodules("torch"),
    *collect_submodules("torchaudio"),
    # Demucs
    *collect_submodules("demucs"),
    # Scientific
    "scipy.signal._spectral_helper",
    "scipy.io.wavfile",
    *collect_submodules("librosa"),
    "soundfile",
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtWidgets",
    "PySide6.QtGui",
    "PySide6.QtMultimedia",
    "PySide6.QtOpenGL",
    # Matplotlib backend
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    # Misc runtime
    "yaml", "tqdm", "einops", "omegaconf", "omegaconf._utils",
    "julius", "antlr4", "treetable", "dora",
]


# ── Analysis ───────────────────────────────────────────────────────────────────

a = Analysis(
    [str(ROOT / "gui.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "_tkinter",
        "PyQt5", "PyQt6",
        "wx", "IPython", "jupyter",
        "pytest", "black", "mypy",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

# ── Platform builds ────────────────────────────────────────────────────────────

if IS_MAC:
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
    app = BUNDLE(  # noqa: F821
        coll,
        name=f"{APP_NAME}.app",
        icon=str(ROOT / "icon.icns") if (ROOT / "icon.icns").exists() else None,
        bundle_identifier="com.trachsplitter.app",
        version=VERSION,
        info_plist={
            "NSPrincipalClass": "NSApplication",
            "NSHighResolutionCapable": True,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHumanReadableCopyright": f"© 2025 TrachSplitter",
        },
    )

elif IS_WIN:
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        icon=str(ROOT / "icon.ico") if (ROOT / "icon.ico").exists() else None,
        version=str(ROOT / "build" / "win_version.txt") if (ROOT / "build" / "win_version.txt").exists() else None,
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )

else:  # Linux
    exe = EXE(  # noqa: F821
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=True,
        upx=True,
        console=False,
        icon=str(ROOT / "icon.png") if (ROOT / "icon.png").exists() else None,
    )
    coll = COLLECT(  # noqa: F821
        exe,
        a.binaries,
        a.datas,
        strip=True,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
