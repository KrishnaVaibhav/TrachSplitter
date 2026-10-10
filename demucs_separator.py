"""Demucs v4 audio separator wrapper."""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile
import logging


logger = logging.getLogger(__name__)


class DemucsSeparator:
    """Wrapper around Demucs v4 for audio source separation."""

    def __init__(
        self,
        model_name: str = 'htdemucs',
        device: str = 'cuda',
        segment: Optional[int] = None,
        overlap: float = 0.25,
        shifts: int = 1,
        progress_callback: Optional[Callable] = None
    ):
        """Initialize Demucs separator.

        Args:
            model_name: Model to use (htdemucs, mdx, etc.)
            device: 'cuda' or 'cpu'
            segment: Segment length in seconds (None for auto)
            overlap: Overlap between segments (0-1)
            shifts: Number of prediction shifts (>1 slower but better)
            progress_callback: Function to call for progress updates
        """
        self.model_name = model_name
        self.device = self._validate_device(device)
        self.segment = segment
        self.overlap = overlap
        self.shifts = shifts
        self.progress_callback = progress_callback
        self.model = None
        self.sources = None
        self._load_model()

    def _validate_device(self, device: str) -> str:
        """Validate and set device."""
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return 'cpu'
        return device

    def _load_model(self):
        """Load Demucs model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.sources = list(getattr(self.model, "sources", []))
            logger.info(f"Model loaded. Sources: {self.sources}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def separate(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        stem_sources: Optional[List[str]] = None,
        two_stems: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Separate audio into stems.

        Args:
            audio_path: Path to audio file
            output_dir: Directory to save separated stems
            stem_sources: List of stems to extract
            two_stems: Extract only one stem (e.g., 'vocals')

        Returns:
            Dictionary mapping stem names to audio arrays
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            logger.info(f"Separating: {audio_path}")
            if self.progress_callback:
                self.progress_callback("Loading audio...", 0)

            # Load audio
            audio = AudioFile(Path(audio_path))
            audio_data = audio.read(streams=0, channels=2, samplerate=44100)
            if isinstance(audio_data, torch.Tensor):
                audio_tensor = audio_data
            else:
                audio_tensor = torch.from_numpy(audio_data)

            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0).repeat(2, 1)

            if self.progress_callback:
                self.progress_callback("Separating stems...", 10)

            # Perform separation with real per-chunk progress
            mix = audio_tensor.unsqueeze(0)
            stems = self._apply_with_progress(mix)

            # Process stems
            result = {}
            for i, source in enumerate(self.sources):
                if stem_sources and source not in stem_sources:
                    continue

                stem_data = stems[0, i].cpu().numpy()
                if stem_data.ndim == 3:
                    stem_data = stem_data[0]  # Remove batch dimension

                result[source] = stem_data

            if self.progress_callback:
                self.progress_callback("Separation complete", 100)

            # Save stems if output_dir specified
            if output_dir:
                self._save_stems(result, output_dir, two_stems)

            return result

        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

    def _segment_seconds(self) -> float:
        """Segment length in seconds, never exceeding the model's own limit.

        Transformer models (htdemucs*) hard-fail on inputs longer than
        their training length, so a user override can only shrink it.
        """
        model_seg = getattr(self.model, 'segment', None)
        if model_seg is None and hasattr(self.model, 'models'):
            segs = [getattr(m, 'segment', None) for m in self.model.models]
            segs = [float(s) for s in segs if s is not None]
            model_seg = min(segs) if segs else None
        if self.segment:
            sec = float(self.segment)
            return min(sec, float(model_seg)) if model_seg else sec
        return float(model_seg) if model_seg else 10.0

    def _apply_with_progress(self, mix: torch.Tensor) -> torch.Tensor:
        """Run apply_model chunk-by-chunk so progress can be reported.

        demucs 4.0.1's apply_model has no progress callback, so we replicate
        its split loop: fixed-length segments with triangular overlap-add
        weighting (transition_power=1), calling the model per segment.

        Args:
            mix: (1, channels, samples) tensor on CPU

        Returns:
            (1, sources, channels, samples) tensor on CPU
        """
        sample_rate = 44100
        seg_len = int(self._segment_seconds() * sample_rate)
        if self.shifts:
            # apply_model's shift trick pads each chunk by up to 0.5 s;
            # leave that headroom or htdemucs rejects the padded length
            seg_len -= int(0.5 * sample_rate)
        seg_len = max(seg_len, sample_rate)
        length = mix.shape[-1]

        def run_chunk(chunk: torch.Tensor) -> torch.Tensor:
            return apply_model(
                self.model,
                chunk.to(self.device),
                shifts=self.shifts,
                split=False,
                overlap=self.overlap,
                progress=False,
                device=self.device,
            ).cpu()

        if length <= seg_len:
            if self.progress_callback:
                self.progress_callback("Separating stems...", 30)
            out = run_chunk(mix)
            if self.progress_callback:
                self.progress_callback("Finalizing...", 95)
            return out

        stride = max(1, int((1 - self.overlap) * seg_len))
        offsets = list(range(0, length, stride))
        # Triangular crossfade weight, same shape demucs uses internally
        weight = torch.cat([
            torch.arange(1, seg_len // 2 + 1),
            torch.arange(seg_len - seg_len // 2, 0, -1),
        ]).float()
        weight = weight / weight.max()

        out = None
        sum_weight = torch.zeros(length)
        for i, offset in enumerate(offsets):
            chunk = mix[..., offset: offset + seg_len]
            chunk_out = run_chunk(chunk)
            if out is None:
                out = torch.zeros(chunk_out.shape[:-1] + (length,))
            chunk_len = chunk_out.shape[-1]
            w = weight[:chunk_len]
            out[..., offset: offset + chunk_len] += w * chunk_out
            sum_weight[offset: offset + chunk_len] += w
            if self.progress_callback:
                pct = 10 + int(85 * (i + 1) / len(offsets))
                self.progress_callback(
                    f"Separating... chunk {i + 1}/{len(offsets)}", pct
                )
        return out / sum_weight.clamp(min=1e-8)

    def _save_stems(
        self,
        stems: Dict[str, np.ndarray],
        output_dir: str,
        two_stems: Optional[str] = None
    ):
        """Save separated stems to disk."""
        os.makedirs(output_dir, exist_ok=True)

        if two_stems:
            # Save requested stem plus the complement mix of all others
            # (e.g. vocals.wav + no_vocals.wav for karaoke)
            stem_data = stems[two_stems]
            torchaudio.save(
                os.path.join(output_dir, f'{two_stems}.wav'),
                torch.from_numpy(stem_data).float(),
                44100
            )
            others = [data for name, data in stems.items() if name != two_stems]
            if others:
                max_len = max(data.shape[-1] for data in others)
                complement = np.zeros((others[0].shape[0], max_len), dtype=np.float32)
                for data in others:
                    complement[:, :data.shape[-1]] += data
                peak = np.abs(complement).max()
                if peak > 1.0:
                    complement /= peak
                torchaudio.save(
                    os.path.join(output_dir, f'no_{two_stems}.wav'),
                    torch.from_numpy(complement).float(),
                    44100
                )
        else:
            # Save all stems
            for stem_name, stem_data in stems.items():
                tensor = torch.from_numpy(stem_data).float()
                if tensor.ndim == 2 and tensor.shape[0] not in (1, 2):
                    tensor = tensor.T
                torchaudio.save(
                    os.path.join(output_dir, f'{stem_name}.wav'),
                    tensor,
                    44100
                )

        logger.info(f"Stems saved to {output_dir}")

    def get_sources(self) -> List[str]:
        """Get available sources for current model."""
        return self.sources or []

    def change_model(self, model_name: str):
        """Change to a different model."""
        self.model_name = model_name
        self._load_model()

    def get_device_info(self) -> Dict[str, str]:
        """Get device information."""
        return {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
            'model': self.model_name,
            'sources': ', '.join(self.get_sources())
        }
