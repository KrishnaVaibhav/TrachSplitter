"""Audio editing and processing utilities."""
import numpy as np
from scipy import signal
from scipy import ndimage
from typing import Dict, List, Optional
import logging


logger = logging.getLogger(__name__)


def _peaking_sos(freq: float, gain_db: float, q: float, fs: int) -> np.ndarray:
    """RBJ cookbook peaking biquad as a single SOS row."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b = [1 + alpha * A, -2 * cos_w0, 1 - alpha * A]
    a = [1 + alpha / A, -2 * cos_w0, 1 - alpha / A]
    return np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]])


def _shelf_sos(freq: float, gain_db: float, fs: int, high: bool, q: float = 0.7071) -> np.ndarray:
    """RBJ cookbook low/high shelf biquad as a single SOS row."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / fs
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    sa = 2 * np.sqrt(A) * alpha
    if high:
        b0 = A * ((A + 1) + (A - 1) * c + sa)
        b1 = -2 * A * ((A - 1) + (A + 1) * c)
        b2 = A * ((A + 1) + (A - 1) * c - sa)
        a0 = (A + 1) - (A - 1) * c + sa
        a1 = 2 * ((A - 1) - (A + 1) * c)
        a2 = (A + 1) - (A - 1) * c - sa
    else:
        b0 = A * ((A + 1) - (A - 1) * c + sa)
        b1 = 2 * A * ((A - 1) - (A + 1) * c)
        b2 = A * ((A + 1) - (A - 1) * c - sa)
        a0 = (A + 1) + (A - 1) * c + sa
        a1 = -2 * ((A - 1) + (A + 1) * c)
        a2 = (A + 1) + (A - 1) * c - sa
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


class AudioEditor:
    """Audio editing and mixing utilities."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio editor.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

    @staticmethod
    def _as_2d(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        return audio.reshape(1, -1) if audio.ndim == 1 else audio

    def mix_stems(
        self,
        stems: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Mix multiple audio stems with optional weighting.

        Args:
            stems: Dictionary mapping stem names to audio arrays
            weights: Dictionary of weights for each stem (default 1.0)
            normalize: Whether to normalize output to prevent clipping

        Returns:
            Mixed audio array (float32)
        """
        if not stems:
            raise ValueError("No stems provided")

        if weights is None:
            weights = {name: 1.0 for name in stems}

        stems = {k: self._as_2d(v) for k, v in stems.items()}
        max_length = max(audio.shape[-1] for audio in stems.values())
        channels = max(audio.shape[0] for audio in stems.values())

        mix = np.zeros((channels, max_length), dtype=np.float32)
        for stem_name, audio in stems.items():
            weight = weights.get(stem_name, 1.0)
            if weight == 0:
                continue
            n = audio.shape[-1]
            # Mono stems broadcast across all output channels
            mix[:, :n] += (audio * weight).astype(np.float32, copy=False)

        if normalize:
            max_val = float(np.abs(mix).max()) if mix.size else 0.0
            if max_val > 1.0:
                mix /= max_val

        return mix

    def adjust_volume(
        self,
        audio: np.ndarray,
        gain_db: float,
        normalize: bool = False
    ) -> np.ndarray:
        """Adjust audio volume by gain in dB.

        Args:
            audio: Audio array
            gain_db: Gain in decibels
            normalize: Whether to normalize to prevent clipping

        Returns:
            Volume-adjusted audio
        """
        gain_linear = 10 ** (gain_db / 20.0)
        adjusted = audio * gain_linear

        if normalize:
            max_val = np.abs(adjusted).max()
            if max_val > 1.0:
                adjusted = adjusted / max_val

        return adjusted

    def apply_eq(
        self,
        audio: np.ndarray,
        frequencies: List[float],
        gains_db: List[float],
        q_factors: Optional[List[float]] = None,
        shelf_edges: bool = False
    ) -> np.ndarray:
        """Apply parametric EQ to audio.

        Args:
            audio: Audio array
            frequencies: List of center frequencies in Hz
            gains_db: List of gains in dB for each frequency
            q_factors: List of Q factors (default 1.0 for all)
            shelf_edges: Use a low shelf for the lowest band and a high
                shelf for the highest (classic graphic-EQ behaviour)

        Returns:
            EQ-adjusted audio
        """
        if len(frequencies) != len(gains_db):
            raise ValueError("Frequencies and gains must have same length")

        if q_factors is None:
            q_factors = [1.0] * len(frequencies)

        audio = self._as_2d(audio)
        lo_f, hi_f = min(frequencies), max(frequencies)
        sections = []
        for freq, gain_db, q in zip(frequencies, gains_db, q_factors):
            if abs(gain_db) < 0.1:  # Skip near-zero gains
                continue
            if freq <= 0 or freq >= self.sample_rate / 2:
                logger.warning(f"Frequency {freq} Hz out of range, skipping")
                continue
            if shelf_edges and len(frequencies) > 1 and freq == lo_f:
                sections.append(_shelf_sos(freq, gain_db, self.sample_rate, high=False))
            elif shelf_edges and len(frequencies) > 1 and freq == hi_f:
                sections.append(_shelf_sos(freq, gain_db, self.sample_rate, high=True))
            else:
                sections.append(_peaking_sos(freq, gain_db, q, self.sample_rate))

        if not sections:
            return audio
        # Single causal pass: filtfilt would apply every band twice (2x the dB)
        sos = np.vstack(sections)
        return signal.sosfilt(sos, audio, axis=-1).astype(np.float32, copy=False)

    def apply_compression(
        self,
        audio: np.ndarray,
        threshold_db: float = -20,
        ratio: float = 4.0,
        attack_ms: float = 10,
        release_ms: float = 100,
        makeup_gain_db: float = 0
    ) -> np.ndarray:
        """Apply dynamic range compression.

        Args:
            audio: Audio array
            threshold_db: Threshold in dB
            ratio: Compression ratio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            makeup_gain_db: Makeup gain in dB

        Returns:
            Compressed audio
        """
        audio = self._as_2d(audio)
        n_samples = audio.shape[1]
        if n_samples == 0:
            return audio

        # Block-based detector (5 ms blocks), vectorized — the previous
        # per-sample Python loop froze the UI for minutes on full songs.
        hop = max(1, int(self.sample_rate * 0.005))
        n_blocks = int(np.ceil(n_samples / hop))
        peak = np.abs(audio).max(axis=0)
        if n_blocks * hop > n_samples:
            peak = np.pad(peak, (0, n_blocks * hop - n_samples))
        block_db = 20 * np.log10(peak.reshape(n_blocks, hop).max(axis=1) + 1e-10)

        over = block_db - threshold_db
        gain_reduction = np.where(over > 0, over * (1.0 / ratio - 1.0), 0.0)

        # Asymmetric attack/release smoothing across blocks (stereo-linked)
        attack_coef = np.exp(-hop / max(attack_ms / 1000.0 * self.sample_rate, 1.0))
        release_coef = np.exp(-hop / max(release_ms / 1000.0 * self.sample_rate, 1.0))
        envelope = np.empty(n_blocks)
        prev = 0.0
        for i in range(n_blocks):
            coef = attack_coef if gain_reduction[i] < prev else release_coef
            prev = coef * prev + (1.0 - coef) * gain_reduction[i]
            envelope[i] = prev

        # Interpolate block gains so steps between blocks don't click
        block_centers = np.arange(n_blocks) * hop + hop / 2
        gain_db = np.interp(np.arange(n_samples), block_centers, envelope) + makeup_gain_db
        return (audio * (10 ** (gain_db / 20.0))[None, :]).astype(np.float32, copy=False)

    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7
    ) -> np.ndarray:
        """Apply a smooth algorithmic reverb (convolution with a synthetic
        exponentially-decaying, decorrelated stereo tail).

        Args:
            audio: Audio array
            room_size: Room size parameter (0-1) — controls decay time
            damping: High-frequency damping (0-1)
            wet_level: Wet signal level (0-1)
            dry_level: Dry signal level (0-1)

        Returns:
            Reverb-processed audio
        """
        audio = self._as_2d(audio).astype(np.float32, copy=False)
        if audio.shape[1] == 0:
            return audio
        room_size = float(np.clip(room_size, 0.0, 1.0))
        damping = float(np.clip(damping, 0.0, 1.0))
        sr = self.sample_rate

        decay_sec = 0.25 + 2.75 * room_size
        predelay = int(0.012 * sr)
        n_ir = int(decay_sec * sr)
        t = np.arange(n_ir) / sr
        envelope = np.exp(-6.9 * t / decay_sec)  # -60 dB at decay_sec

        rng = np.random.default_rng(1234)
        cutoff = 1500 + (1.0 - damping) * 12000
        sos = signal.butter(2, min(cutoff, sr / 2 - 100), btype='low', fs=sr, output='sos')

        wet = np.empty_like(audio)
        for ch in range(audio.shape[0]):
            noise = signal.sosfilt(sos, rng.standard_normal(n_ir))
            ir = np.concatenate([np.zeros(predelay), noise * envelope])
            ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
            wet[ch] = signal.oaconvolve(audio[ch], ir.astype(np.float32))[:audio.shape[1]]

        result = audio * dry_level + wet * wet_level
        return result.astype(np.float32, copy=False)

    def normalize(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Peak-normalize audio so its loudest sample sits at target_db dBFS.

        Args:
            audio: Audio array
            target_db: Target peak level in dBFS

        Returns:
            Normalized audio
        """
        peak = float(np.abs(audio).max()) if np.size(audio) else 0.0
        if peak < 1e-9:
            return audio
        return audio * (10 ** (target_db / 20.0) / peak)

    def limit(self, audio: np.ndarray, ceiling_db: float = -1.0, window_ms: float = 5.0) -> np.ndarray:
        """Transparent look-ahead peak limiter (vectorized).

        Gain is the per-sample amount needed to stay under the ceiling,
        widened with a min-filter and smoothed so it never overshoots.
        """
        audio = self._as_2d(audio)
        ceiling = 10 ** (ceiling_db / 20.0)
        peak = np.abs(audio).max(axis=0)
        if peak.size == 0 or peak.max() <= ceiling:
            return audio
        w = max(1, int(window_ms / 1000.0 * self.sample_rate))
        needed = np.minimum(1.0, ceiling / (peak + 1e-12))
        gain = ndimage.minimum_filter1d(needed, size=2 * w + 1)
        gain = ndimage.uniform_filter1d(gain, size=w)
        return (audio * gain[None, :]).astype(np.float32, copy=False)

    def normalize_loudness(
        self,
        audio: np.ndarray,
        target_lufs: float = -14.0,
        ceiling_db: float = -1.0
    ) -> np.ndarray:
        """Normalize to an integrated loudness (LUFS, ITU-R BS.1770),
        then limit peaks so the result never clips.

        -14 LUFS matches Spotify/YouTube; -16 Apple Music; -23 broadcast.
        """
        from audio_analysis import integrated_loudness
        audio = self._as_2d(audio)
        current = integrated_loudness(audio, self.sample_rate)
        if not np.isfinite(current):
            return audio
        gain_db = target_lufs - current
        out = audio
        # Limiting removes some loudness; re-measure and push up to 3 times,
        # capped so dense material isn't crushed by more than +6 dB extra
        for _ in range(3):
            out = self.limit(audio * (10 ** (gain_db / 20.0)), ceiling_db)
            err = target_lufs - integrated_loudness(out, self.sample_rate)
            if not np.isfinite(err) or abs(err) < 0.3:
                break
            gain_db = min(gain_db + err, target_lufs - current + 6.0)
        return out

    def fade_in(self, audio: np.ndarray, duration_ms: float) -> np.ndarray:
        """Apply fade in.

        Args:
            audio: Audio array
            duration_ms: Fade duration in milliseconds

        Returns:
            Audio with fade in
        """
        audio = self._as_2d(audio)
        fade_samples = min(int(duration_ms * self.sample_rate / 1000), audio.shape[1])
        if fade_samples <= 0:
            return audio
        result = audio.copy()
        result[:, :fade_samples] *= np.linspace(0, 1, fade_samples)
        return result

    def fade_out(self, audio: np.ndarray, duration_ms: float) -> np.ndarray:
        """Apply fade out.

        Args:
            audio: Audio array
            duration_ms: Fade duration in milliseconds

        Returns:
            Audio with fade out
        """
        audio = self._as_2d(audio)
        fade_samples = min(int(duration_ms * self.sample_rate / 1000), audio.shape[1])
        if fade_samples <= 0:
            return audio
        result = audio.copy()
        result[:, -fade_samples:] *= np.linspace(1, 0, fade_samples)
        return result

    def reduce_noise(
        self,
        audio: np.ndarray,
        strength: float = 0.5,
        nperseg: int = 2048
    ) -> np.ndarray:
        """Reduce broadband noise with spectral gating (no extra deps).

        Estimates a per-frequency noise floor from the quietest frames,
        then attenuates time-frequency bins near or below that floor.

        Args:
            audio: Audio array (channels x samples)
            strength: 0 (off) to 1 (aggressive gating)
            nperseg: STFT window size

        Returns:
            De-noised audio, same shape
        """
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == 0.0:
            return audio
        audio = self._as_2d(audio)

        n_samples = audio.shape[1]
        result = np.zeros_like(audio)
        for ch in range(audio.shape[0]):
            f, t, Z = signal.stft(audio[ch], fs=self.sample_rate, nperseg=nperseg)
            mag = np.abs(Z)
            # Noise floor: 10th percentile magnitude per frequency bin
            noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
            threshold = noise_floor * (1.0 + 4.0 * strength)
            # Soft Wiener-style mask, smoothed over time to avoid musical noise
            mask = np.maximum(0.0, 1.0 - (threshold / (mag + 1e-10)) ** 2)
            mask = ndimage.uniform_filter1d(mask, size=5, axis=1, mode='nearest')
            _, rec = signal.istft(Z * mask, fs=self.sample_rate, nperseg=nperseg)
            result[ch] = rec[:n_samples] if len(rec) >= n_samples else np.pad(
                rec, (0, n_samples - len(rec))
            )
        return result

    def apply_pan(self, audio: np.ndarray, pan: float) -> np.ndarray:
        """Apply constant-power stereo panning.

        Args:
            audio: Audio array (channels x samples)
            pan: Pan position from -1.0 (full left) to 1.0 (full right)

        Returns:
            Panned stereo audio
        """
        pan = float(np.clip(pan, -1.0, 1.0))
        audio = self._as_2d(audio)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        theta = (pan + 1.0) * np.pi / 4.0
        # sqrt(2) keeps center pan at unity gain
        gains = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32) * np.sqrt(2.0)
        return audio[:2] * gains[:, None]

    def pitch_shift(
        self,
        audio: np.ndarray,
        semitones: float
    ) -> np.ndarray:
        """Shift pitch without changing tempo.

        Args:
            audio: Audio array
            semitones: Number of semitones to shift

        Returns:
            Pitch-shifted audio
        """
        import librosa
        audio = self._as_2d(audio)
        if abs(semitones) < 1e-3:
            return audio
        # librosa processes multichannel arrays along the last axis in one call
        return librosa.effects.pitch_shift(
            np.ascontiguousarray(audio, dtype=np.float32),
            sr=self.sample_rate, n_steps=semitones
        )

    def time_stretch(
        self,
        audio: np.ndarray,
        rate: float
    ) -> np.ndarray:
        """Change tempo without changing pitch.

        Args:
            audio: Audio array
            rate: Time stretch factor (>1 = faster, <1 = slower)

        Returns:
            Time-stretched audio
        """
        import librosa
        audio = self._as_2d(audio)
        if rate <= 0:
            raise ValueError("Tempo rate must be positive")
        if abs(rate - 1.0) < 1e-3:
            return audio
        return librosa.effects.time_stretch(
            np.ascontiguousarray(audio, dtype=np.float32), rate=rate
        )

    def export_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        audio_format: Optional[str] = None,
        bitrate: str = '192k'
    ) -> str:
        """Export audio to WAV, FLAC, OGG, or MP3.

        Args:
            audio: Audio array (channels, samples)
            output_path: Output file path; extension is adjusted to format
            audio_format: 'wav', 'flac', 'ogg', or 'mp3' (default from extension)
            bitrate: Bitrate for lossy formats

        Returns:
            Path of the written file
        """
        import os
        import soundfile as sf

        audio = self._as_2d(audio)
        # Hard clip: integer formats wrap around on overs, which sounds awful
        audio = np.clip(audio, -1.0, 1.0)

        if audio_format is None:
            audio_format = os.path.splitext(output_path)[1].lstrip('.').lower() or 'wav'
        audio_format = audio_format.lower()

        base, _ = os.path.splitext(output_path)
        output_path = f"{base}.{audio_format}"
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if audio_format in ('wav', 'flac', 'ogg'):
            sf.write(output_path, audio.T, self.sample_rate)
            return output_path

        if audio_format == 'mp3':
            try:
                # libsndfile >= 1.1 supports MP3 directly; map bitrate to
                # its 0..1 compression level (0 = highest quality)
                kbps = int(str(bitrate).rstrip('kK') or 192)
                level = float(np.clip((320 - kbps) / 256.0, 0.0, 1.0))
                sf.write(output_path, audio.T, self.sample_rate,
                         format='MP3', compression_level=level)
                return output_path
            except Exception:
                logger.info("soundfile MP3 write failed, falling back to pydub/ffmpeg")
            from pydub import AudioSegment
            pcm = (audio.T * 32767).astype(np.int16)
            segment = AudioSegment(
                pcm.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=audio.shape[0]
            )
            segment.export(output_path, format='mp3', bitrate=bitrate)
            return output_path

        raise ValueError(f"Unsupported format: {audio_format}")
