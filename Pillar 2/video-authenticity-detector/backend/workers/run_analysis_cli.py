"""CLI entry for isolated video analysis (used by the API server via subprocess)."""
import argparse

import backend.runtime_env  # noqa: F401

from backend.services.video_analysis_pipeline import execute_video_analysis_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pillar 2 video analysis pipeline")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--original-filename", required=True)
    args = parser.parse_args()
    execute_video_analysis_pipeline(args.video_id, args.video_path, args.original_filename)


if __name__ == "__main__":
    main()
