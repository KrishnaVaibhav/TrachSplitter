#!/usr/bin/env python3
"""Command-line interface for TrachSplitter."""
import argparse
import os
import sys
from pathlib import Path
import logging

from config import Config
from demucs_separator import DemucsSeparator
from audio_editor import AudioEditor
from audio_analysis import AudioAnalyzer
import soundfile as sf


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def separate_command(args):
    """Handle separation command."""
    config = Config()

    # Create separator
    separator = DemucsSeparator(
        model_name=args.model,
        device=args.device,
        segment=args.segment if args.segment > 0 else None,
        overlap=args.overlap,
        shifts=args.shifts
    )

    # Separate audio
    print(f"📂 Processing: {args.input}")
    print(f"🎛️  Model: {args.model}")
    print(f"🖥️  Device: {args.device}")

    stems = separator.separate(args.input)

    # Save stems
    output_dir = args.output or os.path.join(Path.cwd(), 'separated', args.model)
    audio_format = getattr(args, 'format', 'wav')

    if audio_format == 'wav':
        separator._save_stems(stems, output_dir, args.two_stems)
    else:
        editor = AudioEditor()
        os.makedirs(output_dir, exist_ok=True)
        names = [args.two_stems] if args.two_stems else list(stems.keys())
        for name in names:
            path = editor.export_audio(
                stems[name],
                os.path.join(output_dir, f'{name}.{audio_format}'),
                audio_format
            )
            print(f"  ✓ {os.path.basename(path)}")

    print(f"✅ Complete! Saved to: {output_dir}")


def karaoke_command(args):
    """Handle karaoke command: instrumental + vocals extraction."""
    separator = DemucsSeparator(
        model_name=args.model,
        device=args.device
    )

    print(f"🎤 Karaoke mode: {args.input}")
    stems = separator.separate(args.input)

    output_dir = args.output or os.path.join(
        Path.cwd(), 'karaoke', Path(args.input).stem
    )
    separator._save_stems(stems, output_dir, two_stems='vocals')

    print(f"✅ Saved: vocals.wav + no_vocals.wav (instrumental) → {output_dir}")


def analyze_command(args):
    """Handle analysis command: BPM, key, loudness."""
    analyzer = AudioAnalyzer()

    print(f"🔍 Analyzing: {args.input}")
    info = analyzer.analyze_file(args.input)

    print(f"  Duration:  {info['duration_sec']} s")
    print(f"  Tempo:     {info['bpm']} BPM")
    print(f"  Key:       {info['key']} {info['mode']} "
          f"(confidence {info['key_confidence']})")
    print(f"  Loudness:  {info['rms_db']} dB RMS, {info['peak_db']} dB peak")

    if args.click:
        click = analyzer.generate_click_track(
            info['beat_times'], info['duration_sec']
        )
        click_path = args.click if isinstance(args.click, str) else 'click.wav'
        sf.write(click_path, click.T, analyzer.sample_rate)
        print(f"🥁 Click track saved: {click_path}")


def mix_command(args):
    """Handle mixing command."""
    editor = AudioEditor()

    # Load stems
    print(f"📂 Loading stems from: {args.stem_dir}")
    stems = {}

    for stem_file in Path(args.stem_dir).glob('*.wav'):
        stem_name = stem_file.stem
        audio, sr = sf.read(str(stem_file))
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        stems[stem_name] = audio
        print(f"  ✓ {stem_name}: {audio.shape}")

    if not stems:
        print("❌ No WAV files found in directory")
        sys.exit(1)

    # Apply gains
    weights = {}
    if args.gains:
        for gain_str in args.gains:
            stem_name, gain_db = gain_str.split(':')
            weights[stem_name] = 10 ** (float(gain_db) / 20.0)
            print(f"  Volume {stem_name}: {gain_db} dB")

    # Mix
    print("🎚️  Mixing stems...")
    mix = editor.mix_stems(stems, weights, normalize=True)

    # Apply effects
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

    if args.normalize:
        print("📈 Normalizing...")
        mix = editor.normalize(mix, target_db=args.normalize_db)

    # Save
    output_path = editor.export_audio(mix, args.output, getattr(args, 'format', None))
    print(f"✅ Mix saved: {output_path}")


def batch_command(args):
    """Handle batch separation."""
    config = Config()

    # Find audio files
    audio_dir = Path(args.input_dir)
    audio_files = list(audio_dir.glob('*.mp3')) + list(audio_dir.glob('*.wav'))
    audio_files += list(audio_dir.glob('*.flac')) + list(audio_dir.glob('*.ogg'))

    if not audio_files:
        print(f"❌ No audio files found in {audio_dir}")
        sys.exit(1)

    print(f"🎵 Found {len(audio_files)} audio files")

    # Separate each
    separator = DemucsSeparator(
        model_name=args.model,
        device=args.device,
        segment=args.segment if args.segment > 0 else None
    )

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")

        try:
            stems = separator.separate(str(audio_file))
            output_dir = args.output_dir / audio_file.stem
            separator._save_stems(stems, str(output_dir))
            print(f"  ✅ Complete")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n✅ Batch processing complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='TrachSplitter - Audio Source Separation & Editing'
    )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Separate command
    sep_parser = subparsers.add_parser('separate', help='Separate audio into stems')
    sep_parser.add_argument('input', help='Input audio file')
    sep_parser.add_argument('-o', '--output', help='Output directory')
    sep_parser.add_argument('-m', '--model', default='htdemucs',
                            choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s',
                                   'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q'],
                            help='Model to use')
    sep_parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'],
                            help='Processing device')
    sep_parser.add_argument('-s', '--segment', type=int, default=0,
                            help='Segment length in seconds (0=auto)')
    sep_parser.add_argument('--overlap', type=float, default=0.25,
                            help='Segment overlap (0-1)')
    sep_parser.add_argument('--shifts', type=int, default=1,
                            help='Number of prediction shifts')
    sep_parser.add_argument('--two-stems', choices=['drums', 'bass', 'vocals', 'other'],
                            help='Extract one stem + complement mix (e.g. vocals + no_vocals)')
    sep_parser.add_argument('-f', '--format', default='wav',
                            choices=['wav', 'flac', 'ogg', 'mp3'],
                            help='Output format')
    sep_parser.set_defaults(func=separate_command)

    # Karaoke command
    kar_parser = subparsers.add_parser(
        'karaoke', help='Extract instrumental (no_vocals) + vocals')
    kar_parser.add_argument('input', help='Input audio file')
    kar_parser.add_argument('-o', '--output', help='Output directory')
    kar_parser.add_argument('-m', '--model', default='htdemucs', help='Model to use')
    kar_parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'],
                            help='Processing device')
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
    mix_parser.add_argument('stem_dir', help='Directory containing stem WAV files')
    mix_parser.add_argument('-o', '--output', default='mix.wav', help='Output file')
    mix_parser.add_argument('-g', '--gains', nargs='*', default=[],
                            help='Stem gains (stem_name:gain_db, e.g., vocals:3 drums:-2)')
    mix_parser.add_argument('--compress', action='store_true', help='Apply compression')
    mix_parser.add_argument('--reverb', action='store_true', help='Apply reverb')
    mix_parser.add_argument('--reverb-room', type=float, default=0.5,
                            help='Reverb room size (0-1)')
    mix_parser.add_argument('--normalize', action='store_true', help='Normalize loudness')
    mix_parser.add_argument('--normalize-db', type=float, default=-3,
                            help='Target loudness in dB')
    mix_parser.add_argument('--pitch', type=float, default=0,
                            help='Pitch shift in semitones (e.g. -2, 1.5)')
    mix_parser.add_argument('--tempo', type=float, default=1.0,
                            help='Tempo factor (>1 = faster, <1 = slower)')
    mix_parser.add_argument('-f', '--format', default=None,
                            choices=['wav', 'flac', 'ogg', 'mp3'],
                            help='Output format (default: from extension)')
    mix_parser.set_defaults(func=mix_command)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch separate multiple files')
    batch_parser.add_argument('input_dir', help='Directory with audio files')
    batch_parser.add_argument('-o', '--output-dir', type=Path, default=Path.cwd() / 'separated',
                              help='Output directory')
    batch_parser.add_argument('-m', '--model', default='htdemucs', help='Model to use')
    batch_parser.add_argument('-d', '--device', default='cuda', choices=['cuda', 'cpu'],
                              help='Processing device')
    batch_parser.add_argument('-s', '--segment', type=int, default=0,
                              help='Segment length in seconds')
    batch_parser.set_defaults(func=batch_command)

    # Parse arguments
    args = parser.parse_args()
    setup_logging(args.verbose)

    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
