"""Configuration management for TrachSplitter."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Application configuration."""

    MODEL_SOURCES = {
        'htdemucs': ['drums', 'bass', 'other', 'vocals'],
        'htdemucs_ft': ['drums', 'bass', 'other', 'vocals'],
        'htdemucs_6s': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
        'hdemucs_mmi': ['drums', 'bass', 'other', 'vocals'],
        'mdx': ['drums', 'bass', 'other', 'vocals'],
        'mdx_extra': ['drums', 'bass', 'other', 'vocals'],
        'mdx_q': ['drums', 'bass', 'other', 'vocals'],
        'mdx_extra_q': ['drums', 'bass', 'other', 'vocals'],
    }

    MODEL_NOTES = {
        'htdemucs': '4 stems, best default quality',
        'htdemucs_ft': '4 stems, slower but highest quality',
        'htdemucs_6s': '6 stems, adds guitar and piano',
        'hdemucs_mmi': '4 stems, stable and fast',
        'mdx': '4 stems, lighter alternative',
        'mdx_extra': '4 stems, improved MDX variant',
        'mdx_q': '4 stems, smaller and faster',
        'mdx_extra_q': '4 stems, smallest fast option',
    }

    MODEL_RECOMMENDATIONS = {
        'htdemucs': 'Recommended: best default',
        'htdemucs_ft': 'Recommended: best quality',
        'htdemucs_6s': 'Recommended: 6 stems',
        'hdemucs_mmi': 'Recommended: balanced',
        'mdx': 'Recommended: lightweight',
        'mdx_extra': 'Recommended: better MDX',
        'mdx_q': 'Recommended: fastest',
        'mdx_extra_q': 'Recommended: fastest small',
    }

    MODEL_GROUPS = {
        'Best Quality': ['htdemucs_ft', 'htdemucs', 'htdemucs_6s'],
        'Balanced': ['hdemucs_mmi', 'mdx_extra', 'mdx'],
        'Fast / Light': ['mdx_q', 'mdx_extra_q'],
    }

    AVAILABLE_MODELS = {
        'htdemucs': 'Hybrid Transformer Demucs (Default) - Best quality',
        'htdemucs_ft': 'Hybrid Transformer Demucs (Fine-tuned) - Slower, better quality',
        'htdemucs_6s': 'Hybrid Transformer Demucs (6 sources) - Adds guitar & piano',
        'hdemucs_mmi': 'Hybrid Demucs v3 - Fast & stable',
        'mdx': 'MDX (MusDB trained) - Alternative model',
        'mdx_extra': 'MDX Extra - Trained with extra data',
        'mdx_q': 'MDX Quantized - Smaller, faster',
        'mdx_extra_q': 'MDX Extra Quantized - Smaller, faster',
    }

    DEFAULT_CONFIG = {
        'model': 'htdemucs',
        'device': None,  # auto-detect GPU first, otherwise CPU
        'segment': None,  # None = auto, otherwise integer seconds
        'overlap': 0.25,
        'shifts': 1,
        'stem_sources': ['drums', 'bass', 'vocals', 'other'],
        'output_format': 'wav',  # wav, mp3, flac
        'output_bitrate': 320,  # for mp3
        'float32': False,
        'int24': False,
        'two_stems': None,  # None or 'vocals', 'drums', 'bass'
        'clip_mode': 'rescale',  # rescale or clamp
        'sample_rate': 44100,
        'batch_processing': True,
        'preview_enabled': True,
        'auto_convert': True,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_path = config_path or os.path.expanduser('~/.trachsplitter/config.json')
        self.config = self.DEFAULT_CONFIG.copy()
        self.load()
        if self.config.get('device') in (None, ''):
            self.config['device'] = self._default_device()

    def load(self):
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except Exception as e:
                print(f"Failed to load config: {e}, using defaults")

    def save(self):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def _default_device(self) -> str:
        """Pick the best available processing device."""
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value

    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
