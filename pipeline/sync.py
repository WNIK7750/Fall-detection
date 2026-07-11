"""
Camera synchronization — estimate per-scenario frame offsets from annotations.

MCFD cameras are not hardware-synchronized; each camera starts recording
independently per scenario. data_tuple3.csv provides frame-level labels that
can be used to estimate the frame offset between cameras.

Approach: for each scenario, compute the median offset of each camera's
fall segment start relative to cam1's fall segment start.
Apply these offsets when reading frames to achieve approximate sync.
"""

import os
import numpy as np
from pipeline.video_reader import load_annotations


def compute_offsets(
    data_root: str,
    scenario: int,
    ref_cam: int = 1,
) -> dict[int, float]:
    """Estimate frame offsets for all cameras relative to a reference camera.

    Uses all fall segments where both the reference camera and target camera
    have a label=1 segment at the same event.

    Returns dict[cam_id] = offset_frames.
    Positive offset means target cam is AHEAD of reference cam
    (e.g., offset=20 means cam5 frame 1103 ≈ cam1 frame 1083,
     so when reading cam1 at frame X, read cam5 at frame X+20).

    Example: Scenario 1, cam1_fall@1083, cam5_fall@1103
      offset[5] = 1103 - 1083 = 20
      When cam1 reads frame 500, cam5 should read frame 500+20=520.
    """
    annotations = load_annotations(data_root)

    # Group fall segments by scenario and camera
    cam_falls: dict[int, list[list[int]]] = {}
    for a in annotations:
        if a["chute"] != scenario or a["label"] != 1:
            continue
        cid = a["cam"]
        if cid not in cam_falls:
            cam_falls[cid] = []
        cam_falls[cid].append([a["start"], a["end"]])

    if ref_cam not in cam_falls:
        return {}

    ref_falls = sorted(cam_falls[ref_cam])
    offsets: dict[int, list[float]] = {}

    for cid, falls in cam_falls.items():
        if cid == ref_cam:
            continue
        target_falls = sorted(falls)

        # Match fall segments by temporal ordering (assume same N events)
        for i, (ref_start, ref_end) in enumerate(ref_falls):
            if i >= len(target_falls):
                break
            tgt_start, _ = target_falls[i]
            # Offset: how many frames is target AHEAD of reference?
            # If target's fall starts at frame 1103 and ref at 1083,
            # target is 20 frames ahead, offset = 20
            off = tgt_start - ref_start
            if cid not in offsets:
                offsets[cid] = []
            offsets[cid].append(off)

    # Use median to be robust against annotation noise
    result: dict[int, float] = {}
    for cid, offs in offsets.items():
        if offs:
            result[cid] = float(np.median(offs))
    return result


def compute_all_offsets(
    data_root: str,
    ref_cam: int = 1,
) -> dict[int, dict[int, float]]:
    """Compute offsets for all scenarios (1-24)."""
    all_offsets = {}
    for s in range(1, 25):
        offs = compute_offsets(data_root, s, ref_cam)
        if offs:
            all_offsets[s] = offs
    return all_offsets


# --- Quick test ---
if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "mcfd")
    root = os.path.abspath(root)

    # Show offsets for first few scenarios
    for s in [1, 2, 4, 5]:
        offs = compute_offsets(root, s, ref_cam=1)
        print(f"Scenario {s:2d} offsets (vs cam1):")
        for cid in sorted(offs):
            print(f"  cam{cid}: {offs[cid]:+.0f} frames")
