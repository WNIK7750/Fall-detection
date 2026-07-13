"""
Review an already-exported short video clip with Qwen3-VL.

Boundary:
  - Camera control / rolling buffer code exports a short clip, e.g. 90 frames.
  - This script only decodes that clip, samples ordered frames, and asks the VL
    model whether the motion shows a fall.

Example:
    python scripts/qa/review_llm_clip.py --video clips/event_001.mp4
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_reviewer import LLMFallReviewer


DEFAULT_MODEL = REPO_ROOT / "models" / "llm" / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = REPO_ROOT / "models" / "llm" / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"


def main():
    parser = argparse.ArgumentParser(description="Review a short fall-event clip with Qwen3-VL")
    parser.add_argument("--video", required=True, help="Path to an already-exported short clip")
    parser.add_argument("--frame-stride", type=int, default=9,
                        help="Sample every Nth frame; 90 frames with stride 9 -> 10 frames")
    parser.add_argument("--max-frames", type=int, default=10,
                        help="Safety cap for sampled frames sent to the VL model")
    parser.add_argument("--no-contact-sheet", action="store_true",
                        help="Send sampled frames separately instead of one timeline contact sheet")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--mmproj", type=str, default=str(DEFAULT_MMPROJ))
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=64)
    parser.add_argument("--llm-image-max-side", type=int, default=0,
                        help="0 disables LLM image resizing")
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    args = parser.parse_args()

    reviewer = LLMFallReviewer(
        model_path=args.model,
        mmproj_path=args.mmproj,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        max_image_side=args.llm_image_max_side,
    )
    if not reviewer.ready:
        raise SystemExit("LLM reviewer failed to load")

    result = reviewer.review_video_file(
        video_path=args.video,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        contact_sheet=not args.no_contact_sheet,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
