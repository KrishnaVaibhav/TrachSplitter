"""Demucs v4 audio separator wrapper."""
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
import torch
import torchaudio
import numpy as np
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile
import logging


logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


class SeparationCancelled(Exception):
    """Raised when a running separation is cancelled by the user."""


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


class DemucsSeparator:
    """Wrapper around Demucs v4 for audio source separation."""

    def __init__(
        self,
        model_name: str = 'htdemucs',
        device: Optional[str] = 'cuda',
        segment: Optional[int] = None,
        overlap: float = 0.25,
        shifts: int = 1,
        progress_callback: Optional[Callable] = None
    ):
        """Initialize Demucs separator.

        Args:
            model_name: Model to use (htdemucs, mdx, etc.)
            device: 'cuda', 'mps' or 'cpu' (None = best available)
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
        self._cancel = threading.Event()
        self._load_model()

    @staticmethod
    def best_device() -> str:
        if torch.cuda.is_available():
            return 'cuda'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return 'mps'
        return 'cpu'

    def _validate_device(self, device: Optional[str]) -> str:
        """Validate and set device."""
        if not device:
            return self.best_device()
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return 'cpu'
        if device == 'mps':
            mps = getattr(torch.backends, 'mps', None)
            if mps is None or not mps.is_available():
                logger.warning("MPS not available, falling back to CPU")
                return 'cpu'
        return device

    def _load_model(self):
        """Load Demucs model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.sources = list(getattr(self.model, "sources", []))
            logger.info(f"Model loaded. Sources: {self.sources}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # ── Cancellation ────────────────────────────────────────────────────────

    def cancel(self):
        """Request cancellation of the running separation."""
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise SeparationCancelled("Separation cancelled")

    def _report(self, msg: str, pct: int):
        if self.progress_callback:
            self.progress_callback(msg, pct)

    # ── Loading ─────────────────────────────────────────────────────────────

    @staticmethod
    def load_audio(audio_path: str) -> torch.Tensor:
        """Load audio as a (2, samples) float tensor at 44.1 kHz.

        Tries FFmpeg (via demucs) first, then soundfile/librosa so WAV,
        FLAC, OGG and MP3 still work on machines without FFmpeg.
        """
        wav = None
        sr = SAMPLE_RATE
        try:
            wav = AudioFile(Path(audio_path)).read(
                streams=0, samplerate=SAMPLE_RATE, channels=2
            )
            if not isinstance(wav, torch.Tensor):
                wav = torch.from_numpy(np.asarray(wav))
        except Exception as e:
            logger.info(f"FFmpeg load failed ({e}); falling back to soundfile")
            try:
                import soundfile as sf
                data, sr = sf.read(audio_path, dtype='float32', always_2d=True)
                wav = torch.from_numpy(np.ascontiguousarray(data.T))
            except Exception as e2:
                logger.info(f"soundfile load failed ({e2}); trying librosa")
                import librosa
                data, sr = librosa.load(audio_path, sr=None, mono=False)
                wav = torch.from_numpy(np.atleast_2d(data).astype(np.float32))

        wav = wav.float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2]
        if wav.shape[-1] == 0:
            raise ValueError("Audio file is empty")
        return wav

    # ── Separation ──────────────────────────────────────────────────────────

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
            two_stems: Return only this stem plus 'no_<stem>' (everything else)

        Returns:
            Dictionary mapping stem names to (channels, samples) float32 arrays
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._cancel.clear()
        logger.info(f"Separating: {audio_path}")
        self._report("Loading audio...", 0)
        wav = self.load_audio(audio_path)
        self._check_cancel()

        # Same input normalization the demucs CLI uses — the models were
        # trained on standardized audio, skipping it costs quality
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
        if not torch.isfinite(std) or std < 1e-8:
            std = torch.tensor(1.0)
        wav = (wav - mean) / std

        self._report("Separating stems...", 5)
        stems = self._apply_with_progress(wav.unsqueeze(0))
        stems = stems * std + mean

        all_stems = {
            source: stems[0, i].numpy().astype(np.float32, copy=False)
            for i, source in enumerate(self.sources)
        }
        del stems

        if two_stems and two_stems in all_stems:
            others = [d for n, d in all_stems.items() if n != two_stems]
            complement = np.sum(others, axis=0).astype(np.float32) if others else \
                np.zeros_like(all_stems[two_stems])
            peak = float(np.abs(complement).max()) if complement.size else 0.0
            if peak > 1.0:
                complement /= peak
            result = {two_stems: all_stems[two_stems], f'no_{two_stems}': complement}
        else:
            result = {
                n: d for n, d in all_stems.items()
                if not stem_sources or n in stem_sources
            }

        self._report("Separation complete", 100)

        if output_dir:
            self._save_stems(result, output_dir)

        return result

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
        seg_len = int(self._segment_seconds() * SAMPLE_RATE)
        if self.shifts:
            # apply_model's shift trick pads each chunk by up to 0.5 s;
            # leave that headroom or htdemucs rejects the padded length
            seg_len -= int(0.5 * SAMPLE_RATE)
        seg_len = max(seg_len, SAMPLE_RATE)
        length = mix.shape[-1]

        def run_chunk(chunk: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
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
            self._report("Separating stems...", 30)
            out = run_chunk(mix)
            self._check_cancel()
            self._report("Finalizing...", 95)
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
        started = time.monotonic()
        for i, offset in enumerate(offsets):
            self._check_cancel()
            chunk = mix[..., offset: offset + seg_len]
            chunk_out = run_chunk(chunk)
            if out is None:
                out = torch.zeros(chunk_out.shape[:-1] + (length,))
            chunk_len = chunk_out.shape[-1]
            w = weight[:chunk_len]
            out[..., offset: offset + chunk_len] += w * chunk_out
            sum_weight[offset: offset + chunk_len] += w
            done = i + 1
            elapsed = time.monotonic() - started
            eta = elapsed / done * (len(offsets) - done)
            pct = 5 + int(90 * done / len(offsets))
            self._report(
                f"Separating... chunk {done}/{len(offsets)} · ~{_format_eta(eta)} left",
                pct
            )
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return out / sum_weight.clamp(min=1e-8)

    def _save_stems(
        self,
        stems: Dict[str, np.ndarray],
        output_dir: str,
        two_stems: Optional[str] = None
    ):
        """Save separated stems to disk as WAV.

        ``two_stems`` is accepted for backward compatibility; ``separate``
        already returns the stem + complement pair when it is requested.
        """
        os.makedirs(output_dir, exist_ok=True)
        if two_stems and f'no_{two_stems}' not in stems and two_stems in stems:
            others = [d for n, d in stems.items() if n != two_stems]
            pair = {two_stems: stems[two_stems]}
            if others:
                pair[f'no_{two_stems}'] = np.sum(others, axis=0)
            stems = pair
        for stem_name, stem_data in stems.items():
            tensor = torch.from_numpy(np.ascontiguousarray(stem_data)).float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            torchaudio.save(
                os.path.join(output_dir, f'{stem_name}.wav'),
                tensor.clamp(-1.0, 1.0),
                SAMPLE_RATE
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
