"""Lightweight visual QA for YOLO 2D pose and triangulated 3D skeletons."""

import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.demo.demo_multiview import draw_3d_skeleton
from pipeline.calibration import get_projections, undistort_keypoints
from pipeline.pose_inference import MultiViewPoseVisualizer, PoseDetector
from pipeline.triangulation import Triangulator
from pipeline.video_reader import MCFDReader, find_fall_segments, load_annotations


def parse_frames(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def sample_frames(
    data_root: str,
    scenario: int,
    ref_cam: int,
    fall_samples: int,
    normal_samples: int,
    seed: int,
) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    annotations = load_annotations(data_root)
    fall_segments = find_fall_segments(annotations, scenario, cam_id=ref_cam)

    frames: list[tuple[int, str]] = []
    for start, end in fall_segments:
        if end <= start:
            continue
        for _ in range(fall_samples):
            frames.append((rng.randint(start, end), "fall"))

    if fall_segments:
        first_fall = min(start for start, _ in fall_segments)
        normal_end = max(1, first_fall - 100)
    else:
        normal_end = 600
    for _ in range(normal_samples):
        frames.append((rng.randint(0, normal_end), "normal"))

    frames.sort(key=lambda item: item[0])
    return frames


def render_sample(
    frames: list[np.ndarray],
    keypoints: list[np.ndarray],
    skeleton_3d: np.ndarray,
    quality: dict,
    camera_ids: list[int],
    label: str,
    frame_idx: int,
) -> np.ndarray:
    vis_frames = []
    for cam_id, frame, kp in zip(camera_ids, frames, keypoints):
        vis = MultiViewPoseVisualizer.draw_keypoints(frame.copy(), kp)
        text = f"cam{cam_id} frame={frame_idx} label={label}"
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vis_frames.append(vis)

    skel_vis = draw_3d_skeleton(np.zeros((220, 220, 3), dtype=np.uint8), skeleton_3d)
    skel_vis = cv2.resize(skel_vis, (vis_frames[0].shape[1], vis_frames[0].shape[0]))
    q_text = (
        f"Q={quality['score']:.2f} valid={quality['valid_joint_ratio']:.2f} "
        f"mv={quality['multiview_joint_ratio']:.2f} err={quality['reproj_error_px']:.1f}px"
    )
    cv2.putText(skel_vis, q_text, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    vis_frames.append(skel_vis)

    return np.hstack(vis_frames)


def main():
    parser = argparse.ArgumentParser(description="Export sampled 2D/3D pose QA images")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "dataset" / "mcfd"))
    parser.add_argument("--scenario", type=int, default=1)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 7])
    parser.add_argument("--frames", default=None, help="Comma-separated reference-camera frames")
    parser.add_argument("--fall-samples", type=int, default=4)
    parser.add_argument("--normal-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "qa_pose_3d"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    explicit_frames = parse_frames(args.frames)
    if explicit_frames:
        samples = [(frame, "manual") for frame in explicit_frames]
    else:
        samples = sample_frames(
            args.data_root,
            args.scenario,
            args.cameras[0],
            args.fall_samples,
            args.normal_samples,
            args.seed,
        )

    detector = PoseDetector(device=args.device)
    tri = Triangulator(get_projections(args.cameras))

    for frame_idx, label in samples:
        with MCFDReader(
            args.data_root,
            args.scenario,
            args.cameras,
            start_frame=frame_idx,
            end_frame=frame_idx + 1,
            undistort=False,
        ) as reader:
            batch = list(reader)
        if not batch:
            print(f"skip frame={frame_idx}: no frames")
            continue

        frames = batch[0]
        keypoints_raw = detector.infer_batch(frames)
        keypoints_undist = [
            undistort_keypoints(kp, args.cameras[v])
            for v, kp in enumerate(keypoints_raw)
        ]
        skeleton_3d = tri.triangulate(keypoints_undist)
        quality = tri.compute_quality(keypoints_undist, skeleton_3d)
        canvas = render_sample(
            frames,
            keypoints_raw,
            skeleton_3d,
            quality,
            args.cameras,
            label,
            frame_idx,
        )

        out_path = os.path.join(
            args.out_dir,
            f"scenario{args.scenario:02d}_frame{frame_idx:05d}_{label}_q{quality['score']:.2f}.jpg",
        )
        cv2.imwrite(out_path, canvas)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
