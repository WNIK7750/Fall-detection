"""
Batch QA helper for Qwen3-VL clip review.

Normal mode loads the VL model once and calls LLMFallReviewer.review_video_file()
for each provided clip. Dry-run mode validates decoding and contact-sheet
sampling without loading model weights.

Examples:
    python scripts/qa/qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 --dry-run
    python scripts/qa/qa_llm_clip_batch.py --videos clips/s01_1050_90.mp4 --output results/llm_qa/clip_batch.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_reviewer import (
    LLMFallReviewer,
    make_contact_sheet,
    sample_video_frames,
)


DEFAULT_MODEL = REPO_ROOT / "models" / "llm" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = REPO_ROOT / "models" / "llm" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"


def dry_run_video(video: str, frame_stride: int, max_frames: int, contact_sheet: bool) -> dict:
    frames = sample_video_frames(video, frame_stride=frame_stride, max_frames=max_frames)
    result = {
        "video_path": video,
        "ok": bool(frames),
        "sampled_frames": len(frames),
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "contact_sheet": contact_sheet,
    }
    if frames:
        result["first_frame_shape"] = list(frames[0].shape)
        if contact_sheet:
            sheet = make_contact_sheet(frames)
            result["contact_sheet_shape"] = list(sheet.shape)
    else:
        result["error"] = "No frames decoded"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test LLM fall clip review")
    parser.add_argument("--videos", nargs="+", required=True, help="Clip files to review")
    parser.add_argument("--frame-stride", type=int, default=9)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Decode/sample only; do not load LLM")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--mmproj", default=str(DEFAULT_MMPROJ))
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--output", default=None, help="Optional JSON output file")
    args = parser.parse_args()

    contact_sheet = not args.no_contact_sheet
    results: list[dict] = []

    if args.dry_run:
        for video in args.videos:
            results.append(dry_run_video(video, args.frame_stride, args.max_frames, contact_sheet))
    else:
        reviewer = LLMFallReviewer(
            model_path=args.model,
            mmproj_path=args.mmproj,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
        )
        if not reviewer.ready:
            raise SystemExit("LLM reviewer failed to load")
        for video in args.videos:
            result = reviewer.review_video_file(
                video_path=video,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
                contact_sheet=contact_sheet,
            )
            results.append(result)

    payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "review_policy": {
            "frame_stride": args.frame_stride,
            "max_frames": args.max_frames,
            "contact_sheet": contact_sheet,
        },
        "results": results,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
