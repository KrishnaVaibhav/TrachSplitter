# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('D:/GitHub/TrachSplitter/config.py', '.'), ('D:/GitHub/TrachSplitter/demucs_separator.py', '.'), ('D:/GitHub/TrachSplitter/audio_editor.py', '.'), ('D:/GitHub/TrachSplitter/audio_analysis.py', '.'), ('D:/GitHub/TrachSplitter/ui_theme.py', '.'), ('D:/GitHub/TrachSplitter/icon.ico', '.'), ('D:/GitHub/TrachSplitter/icon.svg', '.'), ('D:/GitHub/TrachSplitter/venv/Lib/site-packages/matplotlib/mpl-data', 'matplotlib/mpl-data')]
datas += collect_data_files('demucs')
datas += collect_data_files('librosa')


a = Analysis(
    ['D:/GitHub/TrachSplitter/gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['torch', 'torch.nn', 'torch.nn.functional', 'torch.jit', 'torch.utils', 'torch.utils.data', 'torchaudio', 'torchaudio.transforms', 'demucs', 'demucs.apply', 'demucs.audio', 'demucs.htdemucs', 'demucs.hdemucs', 'demucs.pretrained', 'demucs.repo', 'demucs.remote', 'demucs.states', 'demucs.svd', 'demucs.utils', 'demucs.spec', 'demucs.transformer', 'numpy', 'scipy', 'scipy.signal', 'scipy.io', 'scipy.io.wavfile', 'librosa', 'librosa.core', 'librosa.feature', 'librosa.effects', 'soundfile', 'PySide6', 'PySide6.QtCore', 'PySide6.QtWidgets', 'PySide6.QtGui', 'PySide6.QtMultimedia', 'PySide6.QtOpenGL', 'matplotlib', 'matplotlib.figure', 'matplotlib.widgets', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_agg', 'config', 'demucs_separator', 'audio_editor', 'audio_analysis', 'ui_theme', 'scipy.ndimage', 'yaml', 'tqdm', 'einops', 'dora', 'treetable', 'julius', 'antlr4', 'omegaconf', 'omegaconf._utils'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', '_tkinter', 'PyQt5', 'PyQt6', 'wx', 'IPython', 'jupyter', 'notebook', 'pytest', 'black'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TrachSplitter',
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
    version='D:/GitHub/TrachSplitter/build/win_version.txt',
    icon=['D:/GitHub/TrachSplitter/icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TrachSplitter',
)
