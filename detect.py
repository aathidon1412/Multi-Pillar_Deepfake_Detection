"""
================================================================================
Pillar 3 Standalone CLI Voice Deepfake Detector
================================================================================
Thin command-line interface wrapper for acoustic & speech forensics, delegating
to core.pillar3_engine (classify_audio).
"""

import os
import sys
import argparse
import json
from core import classify_audio

def main():
    parser = argparse.ArgumentParser(
        description="Pillar 3: Standalone Voice Deepfake Detector (Real vs Fake)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python detect.py path/to/audio.wav --mode spoken\n  python detect.py path/to/song.mp3 --mode music"
    )
    parser.add_argument("audio_path", type=str, help="Path to the target audio file (.wav, .mp3, etc.)")
    parser.add_argument("--mode", type=str, choices=["spoken", "music"], default="spoken",
                        help="Analysis mode: 'spoken' for conversational speech, 'music' for songs/tracks")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    try:
        result = classify_audio(args.audio_path, mode=args.mode)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("=" * 60)
            print("  VOICE DEEPFAKE DETECTION REPORT")
            print("=" * 60)
            print(f"  File Tested        : {os.path.abspath(args.audio_path)}")
            print(f"  Mode Selected      : {args.mode.upper()}")
            print(f"  Audio Duration     : {result['duration']:.2f}s @ {result['samplerate']} Hz ({result['channels']} ch)")
            print(f"  Music / Beats Check: {'YES (Music/Song Detected)' if result['is_music'] else 'NO (Pure Speech)'}")
            if result.get("is_demixed"):
                print(f"  HPSS Demixing      : Applied (Harmonic Vocal Separated from Percussion)")
            if result.get("disclaimer"):
                print(f"  Notice             : {result['disclaimer']}")
            print("-" * 60)
            print(f"  Verdict            : {result['prediction']}")
            print(f"  Top Confidence     : {result['confidence']:.2f}%")
            print("-" * 60)
            print(f"  [REAL] Human Voice   : {result['real_confidence']:.2f}%")
            print(f"  [FAKE] AI Synthesized: {result['fake_confidence']:.2f}%")
            print("=" * 60)

    except FileNotFoundError as fnf_err:
        print(f"[Error]: {fnf_err}", file=sys.stderr)
        sys.exit(2)
    except ValueError as val_err:
        print(f"[Format Error]: {val_err}", file=sys.stderr)
        sys.exit(3)
    except Exception as ex:
        print(f"[Processing Error]: {ex}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()

