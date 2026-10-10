"""Audio editing and processing utilities."""
import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class AudioEditor:
    """Audio editing and mixing utilities."""

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio editor.

        Args:
            sample_rate: Sample rate in Hz
        """
        self.sample_rate = sample_rate

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
            Mixed audio array
        """
        if not stems:
            raise ValueError("No stems provided")

        if weights is None:
            weights = {name: 1.0 for name in stems}

        # Find reference length
        lengths = [audio.shape[-1] for audio in stems.values()]
        max_length = max(lengths)

        # Mix stems
        mix = np.zeros((stems[list(stems.keys())[0]].shape[0], max_length))
        for stem_name, audio in stems.items():
            weight = weights.get(stem_name, 1.0)
            # Pad audio to max length if needed
            if audio.shape[-1] < max_length:
                padded = np.pad(audio, ((0, 0), (0, max_length - audio.shape[-1])))
            else:
                padded = audio
            mix += padded * weight

        # Normalize if requested
        if normalize:
            max_val = np.abs(mix).max()
            if max_val > 1.0:
                mix = mix / max_val

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
        q_factors: Optional[List[float]] = None
    ) -> np.ndarray:
        """Apply parametric EQ to audio.

        Args:
            audio: Audio array
            frequencies: List of center frequencies in Hz
            gains_db: List of gains in dB for each frequency
            q_factors: List of Q factors (default 1.0 for all)

        Returns:
            EQ-adjusted audio
        """
        if len(frequencies) != len(gains_db):
            raise ValueError("Frequencies and gains must have same length")

        if q_factors is None:
            q_factors = [1.0] * len(frequencies)

        result = audio.copy()

        for freq, gain_db, q in zip(frequencies, gains_db, q_factors):
            if abs(gain_db) < 0.1:  # Skip near-zero gains
                continue

            # Design peaking EQ filter
            nyquist = self.sample_rate / 2
            normalized_freq = freq / nyquist

            if normalized_freq <= 0 or normalized_freq >= 1:
                logger.warning(f"Frequency {freq} Hz out of range, skipping")
                continue

            # Calculate filter coefficients
            omega = 2 * np.pi * normalized_freq
            sin_omega = np.sin(omega)
            alpha = sin_omega / (2 * q)

            A = 10 ** (gain_db / 40.0)

            b0 = 1 + alpha * A
            b1 = -2 * np.cos(omega)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(omega)
            a2 = 1 - alpha / A

            # Normalize
            b = [b0 / a0, b1 / a0, b2 / a0]
            a = [1.0, a1 / a0, a2 / a0]

            # Apply filter to each channel
            for ch in range(result.shape[0]):
                result[ch] = signal.filtfilt(b, a, result[ch])

        return result

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
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        n_samples = audio.shape[1]

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

        gain_db = np.repeat(envelope, hop)[:n_samples] + makeup_gain_db
        return audio * (10 ** (gain_db / 20.0))[None, :]

    def apply_reverb(
        self,
        audio: np.ndarray,
        room_size: float = 0.5,
        damping: float = 0.5,
        wet_level: float = 0.3,
        dry_level: float = 0.7
    ) -> np.ndarray:
        """Apply simple reverb using delays.

        Args:
            audio: Audio array
            room_size: Room size parameter (0-1)
            damping: Damping parameter (0-1)
            wet_level: Wet signal level (0-1)
            dry_level: Dry signal level (0-1)

        Returns:
            Reverb-processed audio
        """
        # Simple comb filter based reverb
        delay_times = [
            int(0.05 * self.sample_rate * room_size),
            int(0.06 * self.sample_rate * room_size),
            int(0.07 * self.sample_rate * room_size),
            int(0.08 * self.sample_rate * room_size),
        ]

        wet = np.zeros_like(audio)

        for delay in delay_times:
            delayed = np.zeros_like(audio)
            delayed[:, delay:] = audio[:, :-delay] if delay < audio.shape[1] else audio[:, :0]
            wet += delayed

        wet = wet / len(delay_times)
        result = audio * dry_level + wet * wet_level

        return result

    def normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target loudness.

        Args:
            audio: Audio array
            target_db: Target loudness in dB

        Returns:
            Normalized audio
        """
        current_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20.0)
        normalized = audio * gain_linear

        # Clip if necessary
        max_val = np.abs(normalized).max()
        if max_val > 1.0:
            normalized = normalized / max_val

        return normalized

    def fade_in(self, audio: np.ndarray, duration_ms: float) -> np.ndarray:
        """Apply fade in.

        Args:
            audio: Audio array
            duration_ms: Fade duration in milliseconds

        Returns:
            Audio with fade in
        """
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, audio.shape[1])

        result = audio.copy()
        fade_curve = np.linspace(0, 1, fade_samples)
        result[:, :fade_samples] *= fade_curve

        return result

    def fade_out(self, audio: np.ndarray, duration_ms: float) -> np.ndarray:
        """Apply fade out.

        Args:
            audio: Audio array
            duration_ms: Fade duration in milliseconds

        Returns:
            Audio with fade out
        """
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, audio.shape[1])

        result = audio.copy()
        fade_curve = np.linspace(1, 0, fade_samples)
        result[:, -fade_samples:] *= fade_curve

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
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

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
            mask = signal.convolve2d(
                mask, np.ones((1, 5)) / 5.0, mode='same', boundary='symm'
            )
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
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        theta = (pan + 1.0) * np.pi / 4.0
        # sqrt(2) keeps center pan at unity gain
        gains = np.array([np.cos(theta), np.sin(theta)]) * np.sqrt(2.0)
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
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        shifted = [
            librosa.effects.pitch_shift(audio[ch], sr=self.sample_rate, n_steps=semitones)
            for ch in range(audio.shape[0])
        ]
        return np.stack(shifted)

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
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        stretched = [
            librosa.effects.time_stretch(audio[ch], rate=rate)
            for ch in range(audio.shape[0])
        ]
        return np.stack(stretched)

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

        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        if audio_format is None:
            audio_format = os.path.splitext(output_path)[1].lstrip('.').lower() or 'wav'
        audio_format = audio_format.lower()

        base, _ = os.path.splitext(output_path)
        output_path = f"{base}.{audio_format}"

        if audio_format in ('wav', 'flac', 'ogg'):
            sf.write(output_path, audio.T, self.sample_rate)
            return output_path

        if audio_format == 'mp3':
            try:
                # libsndfile >= 1.1 supports MP3 directly
                sf.write(output_path, audio.T, self.sample_rate)
                return output_path
            except Exception:
                logger.info("soundfile MP3 write failed, falling back to pydub/ffmpeg")
            from pydub import AudioSegment
            pcm = (np.clip(audio.T, -1.0, 1.0) * 32767).astype(np.int16)
            segment = AudioSegment(
                pcm.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=audio.shape[0]
            )
            segment.export(output_path, format='mp3', bitrate=bitrate)
            return output_path

        raise ValueError(f"Unsupported format: {audio_format}")
