# TrachSplitter API Examples

## Overview

TrachSplitter can be used programmatically in your Python projects for audio separation, mixing, and editing.

## Installation

```bash
pip install trachsplitter
# or
git clone https://github.com/yourusername/TrachSplitter.git
cd TrachSplitter
pip install -e .
```

## Basic Usage

### 1. Simple Audio Separation

```python
from demucs_separator import DemucsSeparator

# Create separator
separator = DemucsSeparator(
    model_name='htdemucs',
    device='cuda'  # or 'cpu'
)

# Separate audio
stems = separator.separate('path/to/song.mp3')

# Access individual stems
drums = stems['drums']
bass = stems['bass']
vocals = stems['vocals']
other = stems['other']

print(f"Separated audio shape: {vocals.shape}")
# Output: (2, 2646000) - stereo, ~60 seconds at 44.1kHz
```

### 2. Save Stems to Disk

```python
import soundfile as sf

# Save individual stems
for stem_name, audio in stems.items():
    sf.write(f'{stem_name}.wav', audio.T, 44100)
    print(f"Saved {stem_name}.wav")
```

### 3. Model Selection

```python
# Change model
separator.change_model('htdemucs_ft')  # Fine-tuned (slower, better)

# Or create with specific model
separator = DemucsSeparator(
    model_name='mdx_q',  # Quantized (faster, smaller)
    device='cuda',
    segment=10,  # Process in 10-second chunks
    shifts=4  # Use 4 prediction shifts for better quality
)

# Get available sources for current model
sources = separator.get_sources()
print(f"Available sources: {sources}")
```

### 4. Progress Callback

```python
def progress_callback(message, progress):
    print(f"[{progress}%] {message}")

separator = DemucsSeparator(
    model_name='htdemucs',
    device='cuda',
    progress_callback=progress_callback
)

stems = separator.separate('song.mp3')
```

## Audio Editing

### 1. Mixing Stems

```python
from audio_editor import AudioEditor

editor = AudioEditor(sample_rate=44100)

# Mix stems with custom weights
weights = {
    'vocals': 1.2,      # Boost vocals by 2dB
    'drums': 1.0,       # Keep drums as-is
    'bass': 0.8,        # Reduce bass by ~2dB
    'other': 0.9        # Slightly reduce other
}

mix = editor.mix_stems(stems, weights, normalize=True)

# Save mix
sf.write('custom_mix.wav', mix.T, 44100)
```

### 2. Volume Control

```python
# Adjust volume of a single stem
vocals_louder = editor.adjust_volume(stems['vocals'], gain_db=3)  # +3 dB
vocals_quiet = editor.adjust_volume(stems['vocals'], gain_db=-6)  # -6 dB
```

### 3. EQ Processing

```python
# Apply parametric EQ to a stem
eq_audio = editor.apply_eq(
    audio=stems['vocals'],
    frequencies=[100, 500, 2000, 5000, 10000],  # 5-band EQ
    gains_db=[2, -1, 3, 1, -2]                   # dB adjustment per band
)

sf.write('vocals_eq.wav', eq_audio.T, 44100)
```

### 4. Compression

```python
# Apply dynamic range compression
compressed = editor.apply_compression(
    audio=stems['vocals'],
    threshold_db=-20,   # Compress above -20dB
    ratio=4.0,          # 4:1 compression ratio
    attack_ms=10,       # 10ms attack
    release_ms=100,     # 100ms release
    makeup_gain_db=2    # Compensate with +2dB
)

sf.write('vocals_compressed.wav', compressed.T, 44100)
```

### 5. Reverb

```python
# Add reverb to drums
reverb_drums = editor.apply_reverb(
    audio=stems['drums'],
    room_size=0.7,      # 0-1, larger = bigger room
    damping=0.5,        # 0-1, higher = less reverb tail
    wet_level=0.4,      # Wet signal level
    dry_level=0.6       # Dry signal level
)

sf.write('drums_reverb.wav', reverb_drums.T, 44100)
```

### 6. Normalization

```python
# Normalize to -3dB loudness
normalized = editor.normalize(stems['vocals'], target_db=-3)

# Normalize with clipping prevention
normalized = editor.apply_compression(
    audio=stems['vocals'],
    threshold_db=-1,
    ratio=10,
    makeup_gain_db=0
)
```

### 7. Fade In/Out

```python
# 3-second fade in
faded_in = editor.fade_in(stems['vocals'], duration_ms=3000)

# 2-second fade out
faded_out = editor.fade_out(stems['vocals'], duration_ms=2000)

# Combine
faded_vocal = editor.fade_in(faded_out, 3000)
```

### 8. Advanced Effects Chain

```python
# Apply effect chain to vocals
vocals = stems['vocals']

# 1. Compress
vocals = editor.apply_compression(vocals, threshold_db=-15, ratio=3)

# 2. EQ
vocals = editor.apply_eq(
    vocals,
    frequencies=[250, 1000, 4000],
    gains_db=[1, 2, 1]
)

# 3. Add reverb
vocals = editor.apply_reverb(vocals, room_size=0.6, wet_level=0.2)

# 4. Normalize
vocals = editor.normalize(vocals, target_db=-3)

# 5. Fade in/out
vocals = editor.fade_in(vocals, 1000)
vocals = editor.fade_out(vocals, 1000)

sf.write('vocals_processed.wav', vocals.T, 44100)
```

## Complete Workflow Example

```python
from demucs_separator import DemucsSeparator
from audio_editor import AudioEditor
import soundfile as sf
import numpy as np

# 1. Separate audio
separator = DemucsSeparator(model_name='htdemucs', device='cuda')
stems = separator.separate('original_song.mp3')

# 2. Process individual stems
editor = AudioEditor()

# Process drums: Add reverb and reduce volume
drums = editor.apply_reverb(stems['drums'], room_size=0.5, wet_level=0.3)
drums = editor.adjust_volume(drums, gain_db=-2)

# Process bass: Compression only
bass = editor.apply_compression(stems['bass'], threshold_db=-18, ratio=2)

# Process vocals: Full chain
vocals = stems['vocals']
vocals = editor.apply_compression(vocals, threshold_db=-20, ratio=3)
vocals = editor.apply_eq(
    vocals,
    frequencies=[150, 1000, 4000, 8000],
    gains_db=[1, 2, 3, 1]
)
vocals = editor.apply_reverb(vocals, room_size=0.6, wet_level=0.2)
vocals = editor.normalize(vocals, target_db=-2)

# Other: Keep mostly unchanged
other = editor.adjust_volume(stems['other'], gain_db=0)

# 3. Create custom mix
processed_stems = {
    'drums': drums,
    'bass': bass,
    'vocals': vocals,
    'other': other
}

weights = {
    'drums': 1.0,
    'bass': 0.9,
    'vocals': 1.1,
    'other': 0.8
}

mix = editor.mix_stems(processed_stems, weights, normalize=True)

# 4. Save result
sf.write('final_mix.wav', mix.T, 44100)
print("✅ Mix created successfully!")
```

## Batch Processing

```python
from pathlib import Path

# Process multiple files
input_dir = Path('music_library')
output_dir = Path('separated_stems')

separator = DemucsSeparator(model_name='htdemucs_ft', device='cuda')

for audio_file in input_dir.glob('*.mp3'):
    print(f"Processing {audio_file.name}...")

    stems = separator.separate(str(audio_file))

    # Save stems in subdirectory
    stem_dir = output_dir / audio_file.stem
    stem_dir.mkdir(parents=True, exist_ok=True)

    for stem_name, audio in stems.items():
        sf.write(
            str(stem_dir / f'{stem_name}.wav'),
            audio.T,
            44100
        )

    print(f"  ✅ Saved to {stem_dir}")
```

## Configuration Management

```python
from config import Config

# Load config
config = Config()

# View settings
print(f"Current model: {config.get('model')}")
print(f"Device: {config.get('device')}")

# Update settings
config.set('model', 'mdx_q')
config.set('device', 'cpu')
config.update({
    'output_format': 'mp3',
    'output_bitrate': 192
})

# Save to disk
config.save()

# Reset to defaults
config.reset_to_defaults()
```

## Device Management

```python
# Check device info
info = separator.get_device_info()
print(f"Device: {info['device']}")
print(f"CUDA: {info['cuda_available']}")
print(f"GPU: {info['cuda_device']}")

# Switch device
if torch.cuda.is_available():
    separator.device = 'cuda'
else:
    separator.device = 'cpu'
```

## Advanced: Custom Audio Processing

```python
import numpy as np

# Work directly with audio arrays
vocals = stems['vocals']  # Shape: (2, num_samples)

# Get mono mix
vocals_mono = np.mean(vocals, axis=0)

# Manual gain adjustment
gain_linear = 10 ** (3 / 20)  # Convert 3dB to linear
vocals_boosted = vocals * gain_linear

# Peak detection
peak = np.abs(vocals).max()
if peak > 1.0:
    # Normalize to prevent clipping
    vocals = vocals / peak

# Energy analysis
rms = np.sqrt(np.mean(vocals ** 2))
db_level = 20 * np.log10(rms)
print(f"Signal level: {db_level:.1f} dB")
```

## CLI Usage

```bash
# Separate audio
python cli.py separate song.mp3 -o stems/ -m htdemucs

# Mix stems
python cli.py mix stems/ -o mixed.wav -g vocals:3 drums:-2 --normalize

# Batch process
python cli.py batch music_dir/ -o separated/ -m mdx_q

# With GPU optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python cli.py separate song.mp3
```

## Performance Tips

### Memory Optimization
```python
# Reduce memory for low-VRAM GPUs
separator = DemucsSeparator(
    model_name='mdx_q',      # Quantized model
    device='cuda',
    segment=8,               # 8-second chunks
    overlap=0.1              # Less overlap
)

# Or set environment variable
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
```

### Speed Optimization
```python
# Fast processing
separator = DemucsSeparator(
    model_name='mdx_q',  # Fastest model
    device='cuda',
    shifts=1             # Single prediction
)

# vs. Quality optimization
separator = DemucsSeparator(
    model_name='htdemucs_ft',  # Best quality
    device='cuda',
    shifts=4,                   # Multiple predictions
    segment=None                # Full audio
)
```

## Error Handling

```python
try:
    separator = DemucsSeparator(model_name='htdemucs', device='cuda')
except RuntimeError as e:
    print(f"GPU initialization failed: {e}")
    separator = DemucsSeparator(model_name='htdemucs', device='cpu')

try:
    stems = separator.separate('song.mp3')
except FileNotFoundError:
    print("Audio file not found!")
except Exception as e:
    print(f"Separation error: {e}")
```

## Integration with Other Libraries

### With librosa
```python
import librosa

# Combine TrachSplitter with librosa
vocals = stems['vocals']
tempo, beats = librosa.beat.beat_track(y=vocals[0], sr=44100)
print(f"Detected tempo: {tempo} BPM")
```

### With music21
```python
from music21 import environment

# Analyze separated audio
# Create MIDI from vocals
# Transpose stems
```

### With pydub
```python
from pydub import AudioSegment
from pydub.effects import normalize

# Convert to pydub for audio effects
mix_pydub = AudioSegment(
    mix.tobytes(),
    frame_rate=44100,
    sample_width=mix.dtype.itemsize,
    channels=2
)

# Apply pydub effects
normalized = normalize(mix_pydub)
```

## Troubleshooting

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Check model download
from demucs.pretrained import get_model
try:
    model = get_model('htdemucs')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model download error: {e}")
```

---

For more examples and documentation, visit the [GitHub repository](https://github.com/yourusername/TrachSplitter).
