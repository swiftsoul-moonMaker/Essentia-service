import argparse
import json
import os
import sys

# Ensure the repo root is on the path when running as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.main import _extract_features  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Essentia feature extraction locally.")
    parser.add_argument("audio_path", help="Path to a local audio file (wav/mp3/flac, etc.)")
    parser.add_argument(
        "--track-id",
        dest="track_id",
        default=None,
        help="Optional track identifier to include in the output.",
    )
    args = parser.parse_args()

    result = _extract_features(args.audio_path, track_id=args.track_id)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
