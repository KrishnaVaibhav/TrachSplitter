#!/usr/bin/env python3
"""Command-line interface for TrachSplitter."""
import argparse
import os
import sys
from pathlib import Path
import logging

from config import Config
from audio_editor import AudioEditor
from audio_analysis import AudioAnalyzer
import soundfile as sf


logger = logging.getLogger(__name__)

MODELS = list(Config.AVAILABLE_MODELS.keys())
STEMS = ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano']
FORMATS = ['wav', 'flac', 'ogg', 'mp3']
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif', '.wma'}


def setup_logging(verbose: bool = False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def _make_separator(args, segment=None, overlap=0.25, shifts=1):
    from demucs_separator import DemucsSeparator

    def progress(msg, pct):
        print(f"\r  [{pct:3d}%] {msg:<60}", end='', flush=True)
        if pct >= 100:
            print()

    return DemucsSeparator(
        model_name=args.model,
        device=args.device,
        segment=segment,
        overlap=overlap,
        shifts=shifts,
        progress_callback=progress,
    )


def _write_stems(stems, output_dir, audio_format, bitrate='320k'):
    editor = AudioEditor()
    os.makedirs(output_dir, exist_ok=True)
    for name, audio in stems.items():
        path = editor.export_audio(
            audio, os.path.join(output_dir, f'{name}.{audio_format}'),
            audio_format, bitrate
        )
        print(f"  ✓ {os.path.basename(path)}")


def separate_command(args):
    """Handle separation command."""
    separator = _make_separator(
        args,
        segment=args.segment if args.segment > 0 else None,
        overlap=args.overlap,
        shifts=args.shifts,
    )

    print(f"📂 Processing: {args.input}")
    print(f"🎛️  Model: {args.model}")
    print(f"🖥️  Device: {separator.device}")

    stems = separator.separate(args.input, two_stems=args.two_stems)

    output_dir = args.output or os.path.join(
        Path.cwd(), 'separated', args.model, Path(args.input).stem
    )
    _write_stems(stems, output_dir, args.format)
    print(f"✅ Complete! Saved to: {output_dir}")


def karaoke_command(args):
    """Handle karaoke command: instrumental + vocals extraction."""
    separator = _make_separator(args)

    print(f"🎤 Karaoke mode: {args.input}")
    stems = separator.separate(args.input, two_stems='vocals')

    output_dir = args.output or os.path.join(
        Path.cwd(), 'karaoke', Path(args.input).stem
    )
    _write_stems(stems, output_dir, args.format)
    print(f"✅ Saved vocals + no_vocals (instrumental) → {output_dir}")


def analyze_command(args):
    """Handle analysis command: BPM, key, loudness."""
    analyzer = AudioAnalyzer()

    print(f"🔍 Analyzing: {args.input}")
    info = analyzer.analyze_file(args.input)

    print(f"  Duration:  {info['duration_sec']} s")
    print(f"  Tempo:     {info['bpm']} BPM")
    print(f"  Key:       {info['key']} {info['mode']} "
          f"(confidence {info['key_confidence']})")
    lufs = f"{info['lufs']} LUFS, " if info.get('lufs') is not None else ""
    print(f"  Loudness:  {lufs}{info['rms_db']} dB RMS, {info['peak_db']} dB peak")

    if args.click:
        click = analyzer.generate_click_track(
            info['beat_times'], info['duration_sec']
        )
        sf.write(args.click, click.T, analyzer.sample_rate)
        print(f"🥁 Click track saved: {args.click}")


def mix_command(args):
    """Handle mixing command."""
    editor = AudioEditor()

    print(f"📂 Loading stems from: {args.stem_dir}")
    stems = {}
    for stem_file in sorted(Path(args.stem_dir).iterdir()):
        if stem_file.suffix.lower() not in ('.wav', '.flac', '.ogg', '.mp3'):
            continue
        audio, sr = sf.read(str(stem_file), dtype='float32', always_2d=True)
        if sr != editor.sample_rate:
            print(f"  ⚠ {stem_file.name}: {sr} Hz (expected {editor.sample_rate}), skipped")
            continue
        stems[stem_file.stem] = audio.T  # soundfile gives (samples, channels)
        print(f"  ✓ {stem_file.stem}: {audio.shape[1]} ch, {audio.shape[0] / sr:.1f} s")

    if not stems:
        print("❌ No audio files found in directory")
        sys.exit(1)

    weights = {}
    for gain_str in args.gains or []:
        try:
            stem_name, gain_db = gain_str.rsplit(':', 1)
            weights[stem_name] = 10 ** (float(gain_db) / 20.0)
        except ValueError:
            print(f"❌ Bad gain '{gain_str}' — use stem:dB, e.g. vocals:-3")
            sys.exit(1)
        if stem_name not in stems:
            print(f"  ⚠ No stem named '{stem_name}' (have: {', '.join(stems)})")
        print(f"  Volume {stem_name}: {gain_db} dB")
    for name in args.mute or []:
        weights[name] = 0.0
        print(f"  Muted {name}")

    print("🎚️  Mixing stems...")
    mix = editor.mix_stems(stems, weights, normalize=True)

    if args.compress:
        print("📊 Applying compression...")
        mix = editor.apply_compression(mix)

    if args.reverb:
        print("🏛️  Applying reverb...")
        mix = editor.apply_reverb(mix, room_size=args.reverb_room)

    if args.pitch:
        print(f"🎹 Pitch shift: {args.pitch:+g} semitones...")
        mix = editor.pitch_shift(mix, args.pitch)

    if args.tempo and args.tempo != 1.0:
        print(f"⏱️  Tempo: x{args.tempo}...")
        mix = editor.time_stretch(mix, args.tempo)

    if args.lufs is not None:
        print(f"📈 Loudness-normalizing to {args.lufs} LUFS...")
        mix = editor.normalize_loudness(mix, target_lufs=args.lufs)
    elif args.normalize:
        print("📈 Normalizing...")
        mix = editor.normalize(mix, target_db=args.normalize_db)

    output_path = editor.export_audio(mix, args.output, args.format)
    print(f"✅ Mix saved: {output_path}")


def batch_command(args):
    """Handle batch separation."""
    audio_dir = Path(args.input_dir)
    pattern = '**/*' if args.recursive else '*'
    audio_files = sorted(
        p for p in audio_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        sys.exit(1)

    print(f"🎵 Found {len(audio_files)} audio files")

    separator = _make_separator(
        args, segment=args.segment if args.segment > 0 else None
    )

    failed = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
        output_dir = args.output_dir / audio_file.stem
        if args.skip_existing and output_dir.is_dir() and any(output_dir.iterdir()):
            print("  ↷ Skipped (output exists)")
            continue
        try:
            stems = separator.separate(str(audio_file), two_stems=args.two_stems)
            _write_stems(stems, str(output_dir), args.format)
        except KeyboardInterrupt:
            print("\n⛔ Interrupted")
            sys.exit(130)
        except Exception as e:
            failed += 1
            print(f"\n  ❌ Error: {e}")

    print(f"\n✅ Batch complete: {len(audio_files) - failed} ok, {failed} failed")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='TrachSplitter - Audio Source Separation & Editing'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    device_kw = dict(default=None, choices=['cuda', 'mps', 'cpu'],
                     help='Processing device (default: best available)')

    # Separate command
    sep_parser = subparsers.add_parser('separate', help='Separate audio into stems')
    sep_parser.add_argument('input', help='Input audio file')
    sep_parser.add_argument('-o', '--output', help='Output directory')
    sep_parser.add_argument('-m', '--model', default='htdemucs', choices=MODELS,
                            help='Model to use')
    sep_parser.add_argument('-d', '--device', **device_kw)
    sep_parser.add_argument('-s', '--segment', type=int, default=0,
                            help='Segment length in seconds (0=auto)')
    sep_parser.add_argument('--overlap', type=float, default=0.25,
                            help='Segment overlap (0-1)')
    sep_parser.add_argument('--shifts', type=int, default=1,
                            help='Number of prediction shifts')
    sep_parser.add_argument('--two-stems', choices=STEMS,
                            help='Extract one stem + complement mix (e.g. vocals + no_vocals)')
    sep_parser.add_argument('-f', '--format', default='wav', choices=FORMATS,
                            help='Output format')
    sep_parser.set_defaults(func=separate_command)

    # Karaoke command
    kar_parser = subparsers.add_parser(
        'karaoke', help='Extract instrumental (no_vocals) + vocals')
    kar_parser.add_argument('input', help='Input audio file')
    kar_parser.add_argument('-o', '--output', help='Output directory')
    kar_parser.add_argument('-m', '--model', default='htdemucs', choices=MODELS,
                            help='Model to use')
    kar_parser.add_argument('-d', '--device', **device_kw)
    kar_parser.add_argument('-f', '--format', default='wav', choices=FORMATS,
                            help='Output format')
    kar_parser.set_defaults(func=karaoke_command)

    # Analyze command
    ana_parser = subparsers.add_parser(
        'analyze', help='Detect BPM, key, and loudness')
    ana_parser.add_argument('input', help='Input audio file')
    ana_parser.add_argument('--click', nargs='?', const='click.wav', default=None,
                            help='Also generate beat-synced click track WAV (optional path)')
    ana_parser.set_defaults(func=analyze_command)

    # Mix command
    mix_parser = subparsers.add_parser('mix', help='Mix stems')
    mix_parser.add_argument('stem_dir', help='Directory containing stem audio files')
    mix_parser.add_argument('-o', '--output', default='mix.wav', help='Output file')
    mix_parser.add_argument('-g', '--gains', nargs='*', default=[],
                            help='Stem gains (stem_name:gain_db, e.g., vocals:3 drums:-2)')
    mix_parser.add_argument('--mute', nargs='*', default=[],
                            help='Stems to leave out (e.g. --mute vocals)')
    mix_parser.add_argument('--compress', action='store_true', help='Apply compression')
    mix_parser.add_argument('--reverb', action='store_true', help='Apply reverb')
    mix_parser.add_argument('--reverb-room', type=float, default=0.5,
                            help='Reverb room size (0-1)')
    mix_parser.add_argument('--normalize', action='store_true', help='Peak-normalize')
    mix_parser.add_argument('--normalize-db', type=float, default=-1,
                            help='Peak target in dBFS')
    mix_parser.add_argument('--lufs', type=float, default=None,
                            help='Loudness-normalize to this LUFS (e.g. -14 for streaming)')
    mix_parser.add_argument('--pitch', type=float, default=0,
                            help='Pitch shift in semitones (e.g. -2, 1.5)')
    mix_parser.add_argument('--tempo', type=float, default=1.0,
                            help='Tempo factor (>1 = faster, <1 = slower)')
    mix_parser.add_argument('-f', '--format', default=None, choices=FORMATS,
                            help='Output format (default: from extension)')
    mix_parser.set_defaults(func=mix_command)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch separate multiple files')
    batch_parser.add_argument('input_dir', help='Directory with audio files')
    batch_parser.add_argument('-o', '--output-dir', type=Path, default=Path.cwd() / 'separated',
                              help='Output directory')
    batch_parser.add_argument('-m', '--model', default='htdemucs', choices=MODELS,
                              help='Model to use')
    batch_parser.add_argument('-d', '--device', **device_kw)
    batch_parser.add_argument('-s', '--segment', type=int, default=0,
                              help='Segment length in seconds')
    batch_parser.add_argument('--two-stems', choices=STEMS,
                              help='Extract one stem + complement per file')
    batch_parser.add_argument('-f', '--format', default='wav', choices=FORMATS,
                              help='Output format')
    batch_parser.add_argument('-r', '--recursive', action='store_true',
                              help='Include sub-folders')
    batch_parser.add_argument('--skip-existing', action='store_true',
                              help='Skip files whose output folder already has stems')
    batch_parser.set_defaults(func=batch_command)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n⛔ Interrupted")
            sys.exit(130)
        except Exception as e:
            logger.debug("Command failed", exc_info=True)
            print(f"\n❌ {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
