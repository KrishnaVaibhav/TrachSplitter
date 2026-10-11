"""Audio analysis utilities: tempo, key, loudness, click tracks."""
import numpy as np
from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']


def _k_weighting_sos(sample_rate: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting (high shelf + RLB high-pass) for any rate."""
    # Stage 1: high shelf, +4 dB above ~1.68 kHz
    G, Q, fc = 4.0, 1 / np.sqrt(2), 1681.974450955533
    A = 10 ** (G / 40.0)
    w0 = 2 * np.pi * fc / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    sa = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * c + sa),
         -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - sa)]
    a = [(A + 1) - (A - 1) * c + sa,
         2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - sa]
    shelf = [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]
    # Stage 2: high-pass at ~38 Hz
    Q, fc = 0.5003270373238773, 38.13547087602444
    w0 = 2 * np.pi * fc / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
    a = [1 + alpha, -2 * c, 1 - alpha]
    hp = [b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]]
    return np.array([shelf, hp])


def integrated_loudness(audio: np.ndarray, sample_rate: int = 44100) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4, gated).

    Args:
        audio: (channels, samples) or (samples,) array

    Returns:
        Loudness in LUFS, or -inf for silence / audio shorter than 400 ms
    """
    from scipy import signal
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    elif audio.shape[0] > audio.shape[1]:
        audio = audio.T
    audio = audio[:2]
    block = int(0.4 * sample_rate)
    hop = block // 4  # 75 % overlap
    n = audio.shape[1]
    if n < block:
        return float('-inf')

    weighted = signal.sosfilt(_k_weighting_sos(sample_rate), audio, axis=-1)
    # Per-block mean square via cumulative sums (vectorized)
    csum = np.concatenate(
        [np.zeros((weighted.shape[0], 1)), np.cumsum(weighted ** 2, axis=1)], axis=1
    )
    starts = np.arange(0, n - block + 1, hop)
    z = (csum[:, starts + block] - csum[:, starts]) / block  # (ch, blocks)
    power = z.sum(axis=0)
    with np.errstate(divide='ignore'):
        lk = -0.691 + 10 * np.log10(power + 1e-20)

    gated = lk > -70.0
    if not gated.any():
        return float('-inf')
    rel_threshold = -0.691 + 10 * np.log10(power[gated].mean()) - 10.0
    gated &= lk > rel_threshold
    if not gated.any():
        return float('-inf')
    return float(-0.691 + 10 * np.log10(power[gated].mean()))


class AudioAnalyzer:
    """Tempo, key, and loudness analysis for audio files and arrays."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Collapse (channels, samples) or (samples,) to mono float32."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            # Accept both (channels, samples) and (samples, channels)
            if audio.shape[0] > audio.shape[1]:
                audio = audio.T
            audio = audio.mean(axis=0)
        return audio

    def detect_tempo(self, audio: np.ndarray) -> Tuple[float, np.ndarray]:
        """Detect tempo (BPM) and beat positions.

        Args:
            audio: Audio array, mono or (channels, samples)

        Returns:
            Tuple of (bpm, beat_times_seconds)
        """
        import librosa
        mono = self._to_mono(audio)
        tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=self.sample_rate)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sample_rate)
        bpm = float(np.atleast_1d(tempo)[0])
        return bpm, beat_times

    def detect_key(self, audio: np.ndarray) -> Tuple[str, str, float]:
        """Detect musical key using chroma features and
        Krumhansl-Schmuckler profile correlation.

        Args:
            audio: Audio array, mono or (channels, samples)

        Returns:
            Tuple of (key_name, mode, confidence) e.g. ('A', 'minor', 0.83)
        """
        import librosa
        mono = self._to_mono(audio)
        chroma = librosa.feature.chroma_cqt(y=mono, sr=self.sample_rate)
        chroma_mean = chroma.mean(axis=1)

        best_key, best_mode, best_corr = 'C', 'major', -np.inf
        for shift in range(12):
            rotated = np.roll(chroma_mean, -shift)
            for mode, profile in (('major', _MAJOR_PROFILE), ('minor', _MINOR_PROFILE)):
                corr = np.corrcoef(rotated, profile)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_key = _PITCH_CLASSES[shift]
                    best_mode = mode

        return best_key, best_mode, float(best_corr)

    def measure_loudness(self, audio: np.ndarray) -> Dict[str, float]:
        """Measure RMS and peak loudness in dBFS.

        Args:
            audio: Audio array

        Returns:
            Dict with 'rms_db' and 'peak_db'
        """
        audio = np.asarray(audio, dtype=np.float32)
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.abs(audio).max() if audio.size else 0.0
        return {
            'rms_db': float(20 * np.log10(rms + 1e-10)),
            'peak_db': float(20 * np.log10(peak + 1e-10)),
        }

    def analyze_file(self, audio_path: str) -> Dict:
        """Full analysis of an audio file: tempo, key, loudness, duration.

        Args:
            audio_path: Path to audio file

        Returns:
            Dict with bpm, beat_times, key, mode, key_confidence,
            duration_sec, rms_db, peak_db, sample_rate
        """
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=False)
        audio = np.atleast_2d(audio)
        mono = audio.mean(axis=0)

        bpm, beat_times = self.detect_tempo(mono)
        key, mode, confidence = self.detect_key(mono)
        loudness = self.measure_loudness(mono)
        lufs = integrated_loudness(audio, sr)

        return {
            'lufs': round(lufs, 1) if np.isfinite(lufs) else None,
            'bpm': round(bpm, 1),
            'beat_times': beat_times,
            'key': key,
            'mode': mode,
            'key_confidence': round(confidence, 3),
            'duration_sec': round(len(mono) / sr, 2),
            'sample_rate': sr,
            **{k: round(v, 2) for k, v in loudness.items()},
        }

    def generate_click_track(
        self,
        beat_times: np.ndarray,
        duration_sec: float,
        click_freq: float = 1000.0,
        click_duration: float = 0.05,
        gain: float = 0.7,
    ) -> np.ndarray:
        """Generate metronome click track aligned to detected beats.

        Args:
            beat_times: Beat positions in seconds
            duration_sec: Total track duration in seconds
            click_freq: Click tone frequency in Hz
            click_duration: Length of each click in seconds
            gain: Click amplitude (0-1)

        Returns:
            Stereo audio array (2, samples)
        """
        import librosa
        mono = librosa.clicks(
            times=beat_times,
            sr=self.sample_rate,
            click_freq=click_freq,
            click_duration=click_duration,
            length=int(duration_sec * self.sample_rate),
        ) * gain
        return np.stack([mono, mono])
