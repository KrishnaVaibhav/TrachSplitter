#!/usr/bin/env python3
"""
Cross-platform build script for TrachSplitter.
Produces a standalone directory-based executable for the current OS.

Usage:
    python build.py                  # build release
    python build.py --clean          # wipe build/ dist/ then build
    python build.py --onefile        # single-file exe (slow startup, useful for distribution)
    python build.py --no-package     # skip zip/dmg packaging step

Outputs (inside dist/):
    Windows  ->  dist/TrachSplitter/TrachSplitter.exe   (or TrachSplitter.exe if --onefile)
    macOS    ->  dist/TrachSplitter.app  (+  dist/TrachSplitter-<ver>-mac.dmg)
    Linux    ->  dist/TrachSplitter/TrachSplitter        (+ dist/TrachSplitter-<ver>-linux.tar.gz)

Notes:
    - PyInstaller must be built on the TARGET platform (no cross-compilation).
    - Demucs AI models are NOT bundled; they download on first run to ~/.cache/torch/hub/.
    - The resulting bundle is large (~1-2 GB) due to PyTorch.
    - Run inside your venv so the right packages are found.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

APP_NAME = "TrachSplitter"
VERSION = "1.0.0"
ROOT = Path(__file__).parent.resolve()
DIST = ROOT / "dist"
BUILD_DIR = ROOT / "build"
ENTRY = ROOT / "gui.py"
SPEC_FILE = ROOT / f"{APP_NAME}.spec"

SYSTEM = platform.system()   # "Windows" | "Darwin" | "Linux"


# ── Hidden imports needed by PyInstaller ──────────────────────────────────────

HIDDEN_IMPORTS = [
    # Torch / torchaudio
    "torch", "torch.nn", "torch.nn.functional", "torch.jit", "torch.utils",
    "torch.utils.data", "torchaudio", "torchaudio.transforms",
    # Demucs internals (dynamic imports)
    "demucs", "demucs.api", "demucs.apply", "demucs.audio", "demucs.htdemucs",
    "demucs.hdemucs", "demucs.model", "demucs.pretrained", "demucs.repo",
    "demucs.remote", "demucs.states", "demucs.svd", "demucs.utils",
    "demucs.spec", "demucs.transformer",
    # Scientific stack
    "numpy", "scipy", "scipy.signal", "scipy.signal._spectral_helper",
    "scipy.io", "scipy.io.wavfile",
    "librosa", "librosa.core", "librosa.feature", "librosa.effects",
    "soundfile", "sndfile",
    # PySide6
    "PySide6", "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui",
    "PySide6.QtMultimedia", "PySide6.QtOpenGL",
    # Matplotlib
    "matplotlib", "matplotlib.figure", "matplotlib.widgets",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    # Other runtime dependencies
    "yaml", "tqdm", "einops", "dora", "treetable",
    "julius", "antlr4",
    "omegaconf", "omegaconf._utils",
    "hydra", "hydra._internal",
]

EXCLUDES = [
    "tkinter", "_tkinter", "PyQt5", "PyQt6", "wx",
    "IPython", "jupyter", "notebook", "pytest", "black",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kwargs):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    subprocess.run(cmd, check=True, **kwargs)


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found — installing...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"])


def collect_datas() -> list[tuple[str, str]]:
    """Return (src_glob, dest_dir) pairs for extra data files."""
    datas = []

    # Local project modules that PyInstaller may miss
    for name in ["config.py", "demucs_separator.py", "audio_editor.py"]:
        src = ROOT / name
        if src.exists():
            datas.append((str(src), "."))

    # soundfile binary data (libsndfile)
    try:
        import soundfile as sf
        sf_data = Path(sf.__file__).parent / "_sndfile.cpython*"
        # soundfile includes its library; PyInstaller usually finds it but be explicit
    except ImportError:
        pass

    # matplotlib style data
    try:
        import matplotlib
        mpl_data = Path(matplotlib.__file__).parent / "mpl-data"
        if mpl_data.exists():
            datas.append((str(mpl_data), "matplotlib/mpl-data"))
    except ImportError:
        pass

    return datas


def convert_icon():
    """
    Convert icon.svg → platform icon format.
    Requires: cairosvg (pip install cairosvg) + Pillow (already in deps).
    Falls back silently if cairosvg is not installed.
    """
    svg = ROOT / "icon.svg"
    if not svg.exists():
        return

    try:
        import cairosvg
        from PIL import Image
        import io
    except ImportError:
        print("[icon] cairosvg or Pillow not installed — skipping icon conversion.")
        print("       Run:  pip install cairosvg   to enable icon embedding.")
        return

    sizes = [16, 32, 48, 64, 128, 256, 512]

    def svg_to_pil(size: int) -> "Image.Image":
        png_bytes = cairosvg.svg2png(url=str(svg), output_width=size, output_height=size)
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    if SYSTEM == "Windows":
        out = ROOT / "icon.ico"
        imgs = [svg_to_pil(s) for s in [16, 32, 48, 64, 128, 256]]
        imgs[0].save(out, format="ICO", sizes=[(s, s) for s in [16, 32, 48, 64, 128, 256]],
                     append_images=imgs[1:])
        print(f"[icon] {out}")

    elif SYSTEM == "Darwin":
        iconset = BUILD_DIR / "TrachSplitter.iconset"
        iconset.mkdir(parents=True, exist_ok=True)
        mapping = {
            "icon_16x16.png": 16, "icon_16x16@2x.png": 32,
            "icon_32x32.png": 32, "icon_32x32@2x.png": 64,
            "icon_128x128.png": 128, "icon_128x128@2x.png": 256,
            "icon_256x256.png": 256, "icon_256x256@2x.png": 512,
            "icon_512x512.png": 512,
        }
        for fname, size in mapping.items():
            svg_to_pil(size).save(iconset / fname)
        icns = ROOT / "icon.icns"
        run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)])
        print(f"[icon] {icns}")

    else:  # Linux
        out = ROOT / "icon.png"
        svg_to_pil(256).save(out)
        print(f"[icon] {out}")


def build_pyinstaller(onefile: bool):
    ensure_pyinstaller()

    datas = collect_datas()
    hidden = HIDDEN_IMPORTS

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        "--log-level", "WARN",
    ]

    # Data files
    sep = ";" if SYSTEM == "Windows" else ":"
    for src, dst in datas:
        cmd += ["--add-data", f"{src}{sep}{dst}"]

    # Hidden imports
    for hi in hidden:
        cmd += ["--hidden-import", hi]

    # Excludes
    for ex in EXCLUDES:
        cmd += ["--exclude-module", ex]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Platform-specific flags
    if SYSTEM == "Darwin":
        cmd += [
            "--windowed",          # .app bundle, no terminal window
            "--osx-bundle-identifier", f"com.trachsplitter.app",
        ]
    elif SYSTEM == "Windows":
        cmd += [
            "--windowed",          # no console window
            "--version-file", _write_win_version_file(),
        ]
    # Linux: no --windowed, terminal stays available as fallback

    cmd.append(str(ENTRY))

    run(cmd, cwd=ROOT)


def _write_win_version_file() -> str:
    """Write a Windows version-info file and return its path."""
    parts = VERSION.split(".")
    major, minor, patch = (parts + ["0", "0", "0"])[:3]
    content = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({major}, {minor}, {patch}, 0),
    prodvers=({major}, {minor}, {patch}, 0),
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'TrachSplitter'),
      StringStruct('FileDescription', 'TrachSplitter Audio Separator'),
      StringStruct('FileVersion', '{VERSION}'),
      StringStruct('InternalName', '{APP_NAME}'),
      StringStruct('OriginalFilename', '{APP_NAME}.exe'),
      StringStruct('ProductName', '{APP_NAME}'),
      StringStruct('ProductVersion', '{VERSION}'),
    ])]),
    VarFileInfo([VarStruct('Translation', [0x0409, 1200])])
  ]
)
""".strip()
    path = BUILD_DIR / "win_version.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


# ── Packaging ─────────────────────────────────────────────────────────────────

def package_windows(onefile: bool):
    """Zip up the Windows build."""
    if onefile:
        src = DIST / f"{APP_NAME}.exe"
    else:
        src = DIST / APP_NAME
    if not src.exists():
        print(f"[package] source not found: {src}")
        return
    out = DIST / f"{APP_NAME}-{VERSION}-windows.zip"
    print(f"Packaging → {out}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        if src.is_dir():
            for f in src.rglob("*"):
                zf.write(f, f.relative_to(DIST))
        else:
            zf.write(src, src.name)
    print(f"  Created: {out}  ({out.stat().st_size / 1_048_576:.0f} MB)")


def package_macos():
    """Create a .dmg from the .app bundle using hdiutil."""
    app = DIST / f"{APP_NAME}.app"
    if not app.exists():
        print(f"[package] .app not found: {app}")
        return
    dmg = DIST / f"{APP_NAME}-{VERSION}-mac.dmg"
    print(f"Packaging → {dmg}")
    run([
        "hdiutil", "create",
        "-volname", APP_NAME,
        "-srcfolder", str(app),
        "-ov", "-format", "UDZO",
        str(dmg),
    ])
    print(f"  Created: {dmg}  ({dmg.stat().st_size / 1_048_576:.0f} MB)")


def package_linux(onefile: bool):
    """Create a .tar.gz of the Linux build."""
    if onefile:
        src = DIST / APP_NAME
    else:
        src = DIST / APP_NAME
    if not src.exists():
        print(f"[package] source not found: {src}")
        return
    out = DIST / f"{APP_NAME}-{VERSION}-linux.tar.gz"
    print(f"Packaging → {out}")
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname=APP_NAME)
    print(f"  Created: {out}  ({out.stat().st_size / 1_048_576:.0f} MB)")


def package(onefile: bool):
    if SYSTEM == "Windows":
        package_windows(onefile)
    elif SYSTEM == "Darwin":
        package_macos()
    else:
        package_linux(onefile)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME} standalone executable")
    parser.add_argument("--clean", action="store_true", help="Delete build/ and dist/ before building")
    parser.add_argument("--onefile", action="store_true", help="Single-file executable (slower startup)")
    parser.add_argument("--no-package", action="store_true", help="Skip zip/dmg packaging")
    args = parser.parse_args()

    print(f"Building {APP_NAME} {VERSION} for {SYSTEM} ({platform.machine()})")
    print(f"Python: {sys.version}")
    print(f"Entry:  {ENTRY}")

    if not ENTRY.exists():
        sys.exit(f"ERROR: Entry point not found: {ENTRY}")

    if args.clean:
        for d in (BUILD_DIR, DIST):
            if d.exists():
                print(f"Removing {d}")
                shutil.rmtree(d)

    convert_icon()
    build_pyinstaller(args.onefile)

    if not args.no_package:
        package(args.onefile)

    # Print final location
    if SYSTEM == "Windows":
        result = DIST / (f"{APP_NAME}.exe" if args.onefile else APP_NAME)
    elif SYSTEM == "Darwin":
        result = DIST / f"{APP_NAME}.app"
    else:
        result = DIST / APP_NAME

    print(f"\n{'='*60}")
    print(f"  Build complete!")
    print(f"  Output: {result}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
