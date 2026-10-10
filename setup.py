"""Setup configuration for TrachSplitter."""
from setuptools import setup
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="trachsplitter",
    version="1.0.0",
    description="Professional audio source separation and editing using Demucs v4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="TrachSplitter Contributors",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/TrachSplitter",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
    ],
    python_requires=">=3.8",
    install_requires=[
        "demucs>=4.0.1",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.1",
        "PySide6>=6.5.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    entry_points={
        "console_scripts": [
            "trachsplitter-cli=cli:main",
        ],
        "gui_scripts": [
            "trachsplitter=main:main",
        ],
    },
    # Flat layout: the app is a set of top-level modules, not a package
    py_modules=[
        "main", "gui", "cli", "config", "demucs_separator",
        "audio_editor", "audio_analysis",
    ],
    include_package_data=True,
    keywords=[
        "audio",
        "music",
        "source-separation",
        "demucs",
        "stem-extraction",
        "audio-editing",
        "mixing",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TrachSplitter/issues",
        "Source": "https://github.com/yourusername/TrachSplitter",
        "Documentation": "https://github.com/yourusername/TrachSplitter/wiki",
    },
)
