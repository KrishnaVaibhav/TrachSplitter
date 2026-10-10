# TrachSplitter Quick Start Guide

## 5-Minute Setup

### Step 1: Install (2 min)
```bash
# Clone
git clone https://github.com/yourusername/TrachSplitter.git
cd TrachSplitter

# Install
pip install -r requirements.txt
```

### Step 2: Run (1 min)
```bash
python main.py
```

### Step 3: Separate Audio (2 min)
1. Click "Browse" and select an MP3/WAV file
2. Click "Separate Audio"
3. Wait for completion
4. Click "Save Stems" to download

---

## CLI Quick Start

### Separate a song
```bash
python cli.py separate song.mp3 -o output_folder/
```

### Mix stems with custom gains
```bash
python cli.py mix output_folder/ -g vocals:3 drums:-2 -o final_mix.wav --normalize
```

### Batch process directory
```bash
python cli.py batch music_folder/ -o separated/
```

### Karaoke: instrumental + vocals
```bash
python cli.py karaoke song.mp3 -o karaoke_out/
```

### Analyze BPM, key, loudness (+ optional click track)
```bash
python cli.py analyze song.mp3 --click click.wav
```

### Export stems as MP3/FLAC/OGG
```bash
python cli.py separate song.mp3 -f mp3 -o output_folder/
```

### Pitch shift / tempo change a mix
```bash
python cli.py mix output_folder/ --pitch -2 --tempo 1.1 -o mix.mp3
```

---

## Python API Quick Start

### 3-line separation
```python
from demucs_separator import DemucsSeparator
separator = DemucsSeparator(model_name='htdemucs', device='cuda')
stems = separator.separate('song.mp3')
```

### Mix with custom volumes
```python
from audio_editor import AudioEditor
import soundfile as sf

editor = AudioEditor()
mix = editor.mix_stems(stems, weights={'vocals': 1.2, 'drums': 0.9})
sf.write('mix.wav', mix.T, 44100)
```

---

## Common Issues

### "CUDA out of memory"
```python
# Reduce segment size
separator = DemucsSeparator(model_name='htdemucs', segment=8)
```

### "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### "Slow processing"
```python
# Use quantized model
separator = DemucsSeparator(model_name='mdx_q')  # 2x faster
```

---

## Feature Highlights

| Feature | GUI | CLI | Python |
|---------|-----|-----|--------|
| Separate audio | ✅ | ✅ | ✅ |
| Mix stems | ✅ | ✅ | ✅ |
| EQ / Compression | ✅ | ❌ | ✅ |
| Batch processing | ❌ | ✅ | ✅ |
| Real-time preview | ✅ | ❌ | ❌ |
| Karaoke (instrumental) | ❌ | ✅ | ✅ |
| BPM / key detection | ❌ | ✅ | ✅ |
| Click track (metronome) | ❌ | ✅ | ✅ |
| MP3/FLAC/OGG export | ❌ | ✅ | ✅ |
| Pitch shift / tempo | ❌ | ✅ | ✅ |

---

## Next Steps

1. **Read full docs**: See [README.md](README.md)
2. **API examples**: See [API_EXAMPLES.md](API_EXAMPLES.md)
3. **Settings**: Configure in GUI Settings tab
4. **Troubleshooting**: Check README troubleshooting section

---

## System Requirements

- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **GPU**: Optional (3+ GB VRAM for CUDA)
- **Disk**: 3+ GB for models

---

## Support

- Issues: GitHub Issues
- Questions: GitHub Discussions
- Docs: README.md + API_EXAMPLES.md

Happy separating! 🎵
