# TrachSplitter — Feature Roadmap

## Shipped (v1.0)

- 7 Demucs models, up to 6 stems, karaoke (two-stem) mode
- Stem mixer: per-stem volume / mute / solo, 5-band EQ
- Effects: compression, reverb
- Quick presets + Quick Export from the Separation tab
- Waveform + spectrogram viewer with zoom and drag-to-select
- Batch CLI, GPU (CUDA) acceleration
- Cross-platform PyInstaller builds

## v1.1 — implemented in this iteration

| # | Feature | Notes |
|---|---------|-------|
| 1 | **In-app audio playback** | Play/stop stems and mix previews inside the app (QtMultimedia) — no more bouncing to the system player. Stop button now actually works. |
| 2 | **Click-to-seek + live playhead** | Click the waveform/spectrogram to seek; playhead line tracks playback. |
| 3 | **Drag & drop** | Drop an audio file anywhere on the window to load it. |
| 4 | **Recent files** | Last 10 files, persisted, one-click reload. |
| 5 | **Export selection** | Drag-select a region on the spectrogram and export just that slice of the stem. |
| 6 | **Per-stem pan** | Constant-power pan control per stem in the mixer. |
| 7 | **Fade in/out** | Fade durations applied to previews and exports. |
| 8 | **Normalize to target dB** | Optional loudness normalization on export. |
| 9 | **Output format honored** | Settings-tab format (wav/mp3/flac) + bitrate now actually used by all exports, including per-stem saves. |
| 10 | **Settings persistence** | Device, format, bitrate, float32, working directory survive restarts. |
| 11 | **Master EQ actually applied** | The 5-band EQ sliders previously did nothing; now applied to previews and exports. |
| 12 | **De-noise (spectral gate)** | Pure scipy spectral gating to cut hiss/bleed left by separation — no extra dependencies. |
| 13 | **Track analysis in GUI** | BPM, key (Krumhansl-Schmuckler), loudness shown on the Separation tab (backed by `audio_analysis.py`). |

### v1.1 CLI additions (parallel work)

- `analyze` command: BPM / key / loudness detection, optional beat-synced click track
- `karaoke` command: vocals + no_vocals complement export
- MP3/FLAC/OGG export (`-f`), pitch shift and tempo change on `mix`

## v1.2 — responsiveness & polish

- **New app icon** — original "sound prism" design (`icon.svg` + multi-size `icon.ico`); set on window, taskbar (AppUserModelID), and app.
- **UI freeze fixes** — everything heavy now runs off the UI thread:
  - Compressor rewritten from a per-sample Python loop (minutes of freeze) to a vectorized block detector (~0.2 s for a 3-minute song)
  - Mix preview/export, quick export, selection export, and stem saving all render and encode in worker threads; buttons show a busy state instead of freezing
  - Stem preview WAVs are written in the background after separation — the stem list appears instantly
  - Spectrogram/waveform view is computed off-thread; the waveform draws a 2000-point min/max envelope instead of millions of raw samples
  - Playhead uses matplotlib blitting during playback — no more full-canvas redraws every frame

## v1.3 — responsive UI redesign

Layout rebuilt around standard HCI guidance (8px spacing grid, Fitts's-law
target sizes, F-pattern reading order, Gestalt grouping, visual hierarchy):

- **Responsive window** — sizes to 85% of the screen on first run, 1024×640 minimum, position/size remembered across sessions
- **Separation tab** — master-detail split: scrollable input workflow on the left (file → analysis → model → stems → karaoke → Separate), results (stem list, large expanding spectrogram, playback + export bar) on the right
- **Editing tab** — mixer + presets left, effects + EQ right, both scrollable; transport bar (Preview/Stop/Export) pinned at the bottom so it never scrolls away
- **Primary actions** highlighted (Separate Audio, Preview Mix), secondary actions visually muted; destructive Reset kept small
- **Settings/Info** — single readable ~680px column, scrollable
- **Restyle** — card-style groups, 32px+ interactive targets, visible focus/hover/disabled states, styled scrollbars/tooltips/list rows, splitter handles

## v1.4 — real progress + multi-stem workflow

- **Real separation progress** — demucs 4.0.1 has no progress callback, so the separator now runs the model chunk-by-chunk itself (same triangular overlap-add demucs uses internally; verified bit-transparent) and reports per-chunk progress to the progress bar instead of jumping 0→100.
- **Multi-stem selection** — Ctrl/Shift-click several stems in the output list: the viewer shows the combined mix, Play Stem plays them together as one mix, Export Selection exports the merged time region, and the new **Merge Selected…** button saves any stem combination as a single audio file.

## Backlog (future)

- Cancel button for in-flight separation
- Batch processing tab in the GUI (folder in → stems out)
- Lyric transcription via local Whisper (karaoke text)
- Chord detection with timeline sync (chroma templates)
- Lead/backing vocal split (second separation pass on the vocal stem)
- Pitch shift / time stretch controls in the GUI (CLI already has them)
- Per-stem EQ (current EQ applies to master bus only)
- A/B compare original vs. mix
- Loop playback of a selection
- Waveform overview + minimap navigation
- Export stems as a single multi-track project (e.g. Reaper/Ardour session)
- Loudness metering (LUFS) and clip indicators
