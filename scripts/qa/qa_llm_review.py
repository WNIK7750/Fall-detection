"""
Quick validation for the local Qwen3-VL fall reviewer.

This script samples labeled MCFD frames and asks the VL model whether the
person is falling. It is intentionally independent from the 3D classifier so
we can validate the LLM before wiring it into the suspected-fall path.

Examples:
    python scripts/qa/qa_llm_review.py --scenario 1 --kind both
    python scripts/qa/qa_llm_review.py --scenario 1 --kind fall --frame 1106
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.llm_reviewer import LLMFallReviewer, make_multiview_collage
from pipeline.video_reader import MCFDReader, find_fall_segments, load_annotations


DEFAULT_MODEL = REPO_ROOT / "models" / "llm" / "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"
DEFAULT_MMPROJ = REPO_ROOT / "models" / "llm" / "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"


def _inside_any(frame: int, segments: list[tuple[int, int]]) -> bool:
    return any(start <= frame <= end for start, end in segments)


def _choose_frame(
    data_root: str,
    scenario: int,
    kind: str,
    ref_cam: int,
    explicit_frame: int | None,
) -> int:
    if explicit_frame is not None:
        return explicit_frame

    annotations = load_annotations(data_root)
    falls = find_fall_segments(annotations, scenario, cam_id=ref_cam)

    if kind == "fall":
        if not falls:
            raise ValueError(f"Scenario {scenario} has no fall segment for cam{ref_cam}")
        start, end = falls[0]
        return (start + end) // 2

    # Normal sample: prefer a frame safely before/after the first labeled fall.
    if not falls:
        return 30
    candidates = [max(0, falls[0][0] - 90), falls[-1][1] + 90, 30]
    for frame in candidates:
        if not _inside_any(frame, falls):
            return frame
    raise ValueError(f"Could not find a normal frame for scenario {scenario}")


def _collect_keyframes(
    data_root: str,
    scenario: int,
    cameras: list[int],
    target_frame: int,
    count: int,
    stride: int,
    center_view: int,
    view_mode: str,
) -> list:
    half = count // 2
    wanted = [max(0, target_frame + (i - half) * stride) for i in range(count)]
    start_frame = min(wanted)
    end_frame = max(wanted) + 1
    wanted_set = set(wanted)
    frames = []

    with MCFDReader(
        data_root,
        scenario,
        cameras,
        start_frame=start_frame,
        end_frame=end_frame,
        undistort=False,
    ) as reader:
        for view_frames in reader:
            abs_frame = reader.ref_start_frame + reader.current_frame - 1
            if abs_frame in wanted_set:
                if view_mode == "collage":
                    labels = [f"Cam{cam}" for cam in cameras]
                    frames.append(make_multiview_collage(view_frames, labels=labels))
                else:
                    frames.append(view_frames[center_view].copy())

    if not frames:
        raise RuntimeError(f"No frames collected around frame {target_frame}")
    return frames


def _run_case(args, reviewer: LLMFallReviewer, kind: str) -> dict:
    data_root = str(REPO_ROOT / "dataset" / "mcfd")
    ref_cam = args.cameras[0]
    center_view = min(max(args.center_view, 0), len(args.cameras) - 1)
    frame = _choose_frame(data_root, args.scenario, kind, ref_cam, args.frame)
    frames = _collect_keyframes(
        data_root=data_root,
        scenario=args.scenario,
        cameras=args.cameras,
        target_frame=frame,
        count=args.frames,
        stride=args.stride,
        center_view=center_view,
        view_mode=args.view_mode,
    )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(frames):
            path = save_dir / f"scenario{args.scenario:02d}_{kind}_frame{frame}_k{i}.jpg"
            cv2.imwrite(str(path), img)

    result = reviewer.review(frames)
    return {
        "scenario": args.scenario,
        "kind": kind,
        "target_frame": frame,
        "num_frames": len(frames),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3-VL fall review on MCFD frames")
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--kind", choices=["fall", "normal", "both"], default="both")
    parser.add_argument("--frame", type=int, default=None,
                        help="Absolute reference-camera frame; overrides automatic sampling")
    parser.add_argument("--frames", type=int, default=3, help="Number of keyframes to send")
    parser.add_argument("--stride", type=int, default=8, help="Frame spacing between keyframes")
    parser.add_argument("--center-view", type=int, default=1,
                        help="Camera index within --cameras to send to the VL model")
    parser.add_argument("--view-mode", choices=["collage", "center"], default="collage",
                        help="collage sends all selected cameras in one image; center sends one camera")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--mmproj", type=str, default=str(DEFAULT_MMPROJ))
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=64)
    parser.add_argument("--llm-image-max-side", type=int, default=0,
                        help="0 disables LLM image resizing")
    parser.add_argument("--save-dir", type=str, default="results/llm_qa")
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

    kinds = ["normal", "fall"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        result = _run_case(args, reviewer, kind)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
