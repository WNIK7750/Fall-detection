"""
Export a short MCFD clip for LLM video-review testing.

This is a QA/tooling script only. Production rolling-buffer export should live
in the camera-control layer.

Example:
    python scripts/tools/export_mcfd_clip.py --scenario 1 --start-frame 1050 --frames 90 --output clips/s01_1050_90.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_reviewer import make_multiview_collage
from pipeline.video_reader import MCFDReader


def main():
    parser = argparse.ArgumentParser(description="Export a short MCFD multiview collage clip")
    parser.add_argument("--scenario", type=int, required=True)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output clip FPS; frame content is sampled from MCFD reference timeline")
    args = parser.parse_args()

    data_root = REPO_ROOT / "dataset" / "mcfd"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    frame_count = 0
    with MCFDReader(
        str(data_root),
        args.scenario,
        args.cameras,
        start_frame=args.start_frame,
        end_frame=args.start_frame + args.frames,
        undistort=False,
    ) as reader:
        for frames in reader:
            collage = make_multiview_collage(frames, labels=[f"Cam{cam}" for cam in args.cameras])
            if writer is None:
                h, w = collage.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, args.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open output video: {output}")
            writer.write(collage)
            frame_count += 1

    if writer is not None:
        writer.release()

    print(f"Exported {frame_count} frames to {output}")


if __name__ == "__main__":
    main()
