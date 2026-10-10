#!/usr/bin/env python3
"""TrachSplitter - Local Audio Source Separation & Editing Application.

Uses Meta Research's Demucs v4 for high-quality audio source separation.
Includes comprehensive audio editing and mixing capabilities.
"""
import sys
import os
import logging
from pathlib import Path

# Setup logging
log_dir = Path.home() / '.trachsplitter'
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_dir / 'app.log')
    ]
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all dependencies are installed."""
    required = ['torch', 'torchaudio', 'demucs', 'PySide6', 'numpy', 'scipy', 'librosa']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    logger.info("All dependencies found")
    return True


def main():
    """Main application entry point."""
    try:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        logger.info("Starting TrachSplitter")
        print("TrachSplitter - Audio Source Separation & Editing")
        print("Using Demucs v4 (Meta Research)\n")

        # Import GUI
        from gui import TrachSplitterGUI
        from PySide6.QtWidgets import QApplication

        # Create and show application
        app = QApplication(sys.argv)
        window = TrachSplitterGUI()
        window.show()

        sys.exit(app.exec())

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
