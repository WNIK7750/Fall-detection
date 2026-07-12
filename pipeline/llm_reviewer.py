"""
LLM Fall Review — Qwen3-VL-4B multimodal review for suspected fall events.

Uses llama-cpp-python to run local quantized vision-language model.
Only triggered when classifier outputs "suspected" (threshold_low < P_fall ≤ threshold_high).

Model: Qwen3-VL-4B-Instruct Q4_K_M + mmproj Q8_0 (~3.6GB VRAM)
Input:  1-3 keyframe images from the suspected time window
Output: JSON with fall judgment + confidence + reasoning

Usage:
    reviewer = LLMFallReviewer(
        model_path="models/llm/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        mmproj_path="models/llm/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf",
    )
    if not reviewer.ready:
        print("Model failed to load")
        return
    result = reviewer.review(frames=[img1, img2, img3])
"""

import base64
import json
import re
from typing import Optional

import numpy as np

# Try importing llama-cpp; handle gracefully if not installed
try:
    import llama_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False
    Llama = None
    Qwen25VLChatHandler = None


RAW_FALL_REVIEW_PROMPT = """You are an AI safety reviewer for an elderly fall detection system.

You will be shown {n_frames} ordered frames sampled from a short surveillance
video clip. Use the visible motion over time. Do not assume a fall happened just
because this is a fall detection task. Decide whether the person transitions
into a fall-risk / abnormal collapsed posture OR remains in clearly normal
daily activity.

Output your judgment as a JSON object with exactly these fields:
- "is_fall": true or false (whether a fall-risk event is visible)
- "confidence": a number from 0.0 to 1.0 indicating your certainty
- "reason": a brief explanation in English, 1-2 sentences

IMPORTANT: Only output the JSON object, nothing else. Do not add any other text.

Example output:
{{"is_fall": true, "confidence": 0.92, "reason": "Person is lying on floor."}}
"""


CONTACT_SHEET_REVIEW_PROMPT = """You are an AI safety reviewer for an elderly fall detection system.

You will be shown one contact-sheet image made from {n_frames} ordered frames
sampled from a short surveillance video clip. The tiles are arranged left to
right, top to bottom, and each tile is labeled with its time index. Treat the
contact sheet as a short video timeline.

Judge the motion over time: whether the person transitions from normal/upright
activity into a fall-risk / abnormal collapsed posture, or remains clearly
normal. Focus on the temporal progression, not just one tile.

Output your judgment as a JSON object with exactly these fields:
- "is_fall": true or false
- "confidence": a number from 0.0 to 1.0 indicating your certainty
- "reason": a brief explanation in English, 1-2 sentences

IMPORTANT: Only output the JSON object, nothing else. Do not add any other text.

Example output:
{{"is_fall": true, "confidence": 0.92, "reason": "The sequence shows the person moving from upright posture to lying on the floor."}}
"""


CONTEXT_FALL_REVIEW_PROMPT = """You are an AI safety reviewer for an elderly fall detection system.

You will be shown {n_frames} keyframe images from a multi-camera surveillance video.
These images are shown only after a separate 3D fall classifier has already
flagged the time window as suspicious. Your task is to decide whether the
visual evidence is fall-risk/abnormal posture OR clearly normal daily activity.

The MCFD demo scene uses a white crash mat / soft landing pad for staged falls.
Treat a person who is horizontal, reclined sideways, prone, supine, or motionless
on that mat or on the floor as "is_fall": true. Do not dismiss it as controlled
resting unless the person is clearly sitting upright on a chair/sofa or walking.
Prefer catching a fall-risk event over dismissing a suspicious posture.

Output your judgment as a JSON object with exactly these fields:
- "is_fall": true or false (whether a fall event is occurring)
- "confidence": a number from 0.0 to 1.0 indicating your certainty
- "reason": a brief explanation in English, 1-2 sentences

IMPORTANT: Only output the JSON object, nothing else. Do not add any other text.

Example output:
{{"is_fall": true, "confidence": 0.92, "reason": "Person is lying on floor."}}

Positive examples for "is_fall": true:
- Person is lying sideways or supine on a white floor mat/cushion.
- Person has transitioned from upright to collapsed/reclined posture.

Negative examples for "is_fall": false:
- Person is standing, walking, or sitting upright normally.
- Person is seated upright on a sofa/chair without collapse.

System evidence from YOLO + 3D reconstruction:
{evidence}
"""

SYSTEM_PROMPT = "You are a computer vision safety reviewer. Always respond with valid JSON only."


def make_multiview_collage(
    frames: list[np.ndarray],
    labels: Optional[list[str]] = None,
    tile_size: int = 384,
) -> np.ndarray:
    """Build one compact horizontal image from synchronized camera frames."""
    import cv2

    tiles = []
    for i, frame in enumerate(frames):
        tile = cv2.resize(frame, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        if labels and i < len(labels):
            cv2.rectangle(tile, (0, 0), (90, 26), (0, 0, 0), -1)
            cv2.putText(
                tile,
                labels[i],
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        tiles.append(tile)
    return np.hstack(tiles)


class LLMFallReviewer:
    """Local Qwen3-VL fall event reviewer using llama.cpp."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: Path to GGUF quantized model.
            mmproj_path: Path to mmproj vision projector file.
            n_ctx: Context window size (keep small for low VRAM).
            n_gpu_layers: GPU layers; -1 = all, reduce if OOM.
            verbose: Print llama.cpp logs.
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.ready = False

        if not HAS_LLAMA:
            print("[LLMFallReviewer] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            return

        try:
            if Qwen25VLChatHandler is None:
                raise RuntimeError("Qwen VL chat handler is unavailable in llama-cpp-python")
            chat_handler = Qwen25VLChatHandler(
                clip_model_path=mmproj_path,
                verbose=verbose,
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                chat_handler=chat_handler,
                verbose=verbose,
            )
            self.ready = True
            print(f"[LLMFallReviewer] Loaded: {model_path}")
        except Exception as e:
            print(f"[LLMFallReviewer] Failed to load model: {e}")
            self.llm = None

    def review(
        self,
        frames: list[np.ndarray],
        prompt_text: Optional[str] = None,
        timeout: float = 30.0,
    ) -> dict:
        """Review keyframes for fall event.

        Args:
            frames: BGR images (H×W×3 numpy arrays), 1-3 frames.
            timeout: Max seconds for LLM response.

        Returns:
            dict with keys: is_fall, confidence, reason, raw_response.
            On error: is_fall=None, raw_response="ERROR: ..."
        """
        if not self.ready:
            return _error("Model not loaded")

        images_base64 = []
        for i, frame in enumerate(frames):
            # Convert BGR to RGB for the model
            frame_rgb = frame[..., ::-1]
            # Encode to JPEG base64
            ok, buffer = cv2_imencode(frame_rgb)
            if not ok:
                return _error(f"Failed to encode frame {i}")
            images_base64.append(base64.b64encode(buffer).decode("utf-8"))

        # Build prompt
        if prompt_text is None:
            prompt_text = CONTEXT_FALL_REVIEW_PROMPT.format(
                n_frames=len(frames),
                evidence="No structured evidence was provided.",
            )

        # Create message content with images + text
        content = []
        for img_b64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # Inference
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.1,
                top_p=0.9,
            )
            raw = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return _error(f"LLM inference failed: {e}")

        return _parse_response(raw)

    def review_raw(self, frames: list[np.ndarray]) -> dict:
        """First pass: visual-only review on unmodified frames."""
        return self.review(
            frames,
            prompt_text=RAW_FALL_REVIEW_PROMPT.format(n_frames=len(frames)),
        )

    def review_video_file(
        self,
        video_path: str,
        frame_stride: int = 3,
        max_frames: int | None = 30,
        contact_sheet: bool = False,
    ) -> dict:
        """Review an already-exported short clip.

        The upstream camera/buffer layer owns clip creation. This method only
        decodes the clip, samples ordered frames, and asks the VL model to
        judge the motion.
        """
        frames = sample_video_frames(video_path, frame_stride=frame_stride, max_frames=max_frames)
        if not frames:
            return _error(f"No frames decoded from video: {video_path}")
        if contact_sheet:
            sheet = make_contact_sheet(frames)
            result = self.review(
                [sheet],
                prompt_text=CONTACT_SHEET_REVIEW_PROMPT.format(n_frames=len(frames)),
            )
        else:
            result = self.review_raw(frames)
        result["video_path"] = video_path
        result["sampled_frames"] = len(frames)
        result["frame_stride"] = frame_stride
        result["contact_sheet"] = contact_sheet
        return result

    def review_with_context(self, frames: list[np.ndarray], evidence: str) -> dict:
        """Second pass: review with overlaid system evidence and text summary."""
        return self.review(
            frames,
            prompt_text=CONTEXT_FALL_REVIEW_PROMPT.format(
                n_frames=len(frames),
                evidence=evidence,
            ),
        )

    def review_dual(
        self,
        raw_frames: list[np.ndarray],
        context_frames: list[np.ndarray],
        evidence: str,
        min_confidence: float = 0.6,
    ) -> dict:
        """Run raw visual review, then evidence-assisted review, and OR-combine.

        Final decision:
          - fall if either pass says fall with enough confidence
          - normal if both passes say normal with enough confidence
          - otherwise suspected
        """
        raw = self.review_raw(raw_frames)
        assisted = self.review_with_context(context_frames, evidence)

        raw_fall = raw.get("is_fall") is True and float(raw.get("confidence", 0.0)) >= min_confidence
        assisted_fall = assisted.get("is_fall") is True and float(assisted.get("confidence", 0.0)) >= min_confidence
        raw_normal = raw.get("is_fall") is False and float(raw.get("confidence", 0.0)) >= min_confidence
        assisted_normal = assisted.get("is_fall") is False and float(assisted.get("confidence", 0.0)) >= min_confidence

        if raw_fall or assisted_fall:
            final = "fall"
        elif raw_normal and assisted_normal:
            final = "normal"
        else:
            final = "suspected"

        reason = (
            f"raw={raw.get('is_fall')}({raw.get('confidence', 0.0)}): {raw.get('reason', '')} | "
            f"assisted={assisted.get('is_fall')}({assisted.get('confidence', 0.0)}): {assisted.get('reason', '')}"
        )
        return {
            "is_fall": final == "fall",
            "final_status": final,
            "confidence": max(float(raw.get("confidence", 0.0)), float(assisted.get("confidence", 0.0))),
            "reason": reason[:300],
            "raw_pass": raw,
            "assisted_pass": assisted,
            "raw_response": json.dumps({"raw": raw, "assisted": assisted}, ensure_ascii=False),
        }

    def review_from_trigger(
        self,
        frames_3view: list[list[np.ndarray]],
        center_view: int = 1,
    ) -> dict:
        """Convenience: pick center camera frames for review.

        Args:
            frames_3view: 3 lists of frames from 3 cameras.
            center_view: Which view to use (0/1/2). Default center.
        """
        keyframes = frames_3view[center_view]
        # Take up to 3 evenly-spaced frames
        if len(keyframes) > 3:
            indices = np.linspace(0, len(keyframes) - 1, 3, dtype=int)
            keyframes = [keyframes[i] for i in indices]
        return self.review(keyframes)


def _parse_response(raw: str) -> dict:
    """Extract JSON from LLM response, with fallback heuristics."""
    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        return _validate(parsed, raw)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    m = re.search(r'\{[^{}]*"is_fall"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return _validate(json.loads(m.group()), raw)
        except json.JSONDecodeError:
            pass

    # Heuristic fallback
    raw_lower = raw.lower()
    if any(kw in raw_lower for kw in ["fall", "lying", "collapsed", "ground"]):
        return {"is_fall": True, "confidence": 0.5, "reason": raw[:200], "raw_response": raw}
    return {"is_fall": False, "confidence": 0.5, "reason": raw[:200], "raw_response": raw}


def _validate(parsed: dict, raw: str) -> dict:
    """Validate and normalize parsed result."""
    is_fall = bool(parsed.get("is_fall", False))
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    return {
        "is_fall": is_fall,
        "confidence": round(confidence, 3),
        "reason": str(parsed.get("reason", ""))[:300],
        "raw_response": raw,
    }


def _error(msg: str) -> dict:
    return {"is_fall": None, "confidence": 0.0, "reason": msg, "raw_response": f"ERROR: {msg}"}


def sample_video_frames(
    video_path: str,
    frame_stride: int = 3,
    max_frames: int | None = 30,
) -> list[np.ndarray]:
    """Decode a short video and return ordered sampled BGR frames."""
    import cv2

    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames: list[np.ndarray] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride == 0:
                frames.append(frame)
                if max_frames is not None and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()

    return frames


def make_contact_sheet(
    frames: list[np.ndarray],
    columns: int = 5,
    tile_width: int = 320,
    label_prefix: str = "t",
) -> np.ndarray:
    """Pack ordered frames into one labeled timeline image."""
    import cv2

    if not frames:
        raise ValueError("frames must not be empty")
    columns = max(1, columns)
    rows = int(np.ceil(len(frames) / columns))

    h0, w0 = frames[0].shape[:2]
    tile_height = max(1, int(round(tile_width * h0 / max(w0, 1))))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row, col = divmod(i, columns)
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (72, 24), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{label_prefix}{i:02d}",
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y0 = row * tile_height
        x0 = col * tile_width
        sheet[y0:y0 + tile_height, x0:x0 + tile_width] = tile

    return sheet


# Lazy import for cv2 (used in review())
def cv2_imencode(frame_rgb: np.ndarray) -> tuple:
    import cv2
    return cv2.imencode(".jpg", frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])


# --- Quick test ---
if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(base, "models", "llm", "Qwen3VL-4B-Instruct-Q4_K_M.gguf")
    mmproj = os.path.join(base, "models", "llm", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf")

    print(f"Model: {os.path.exists(model)}")
    print(f"mmproj: {os.path.exists(mmproj)}")

    if os.path.exists(model) and os.path.exists(mmproj):
        reviewer = LLMFallReviewer(model, mmproj, n_gpu_layers=20)
        if reviewer.ready:
            # Test with a dummy black image
            test_img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw text on image so model has something to analyze
            import cv2
            cv2.putText(test_img, "No fall detected", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            print("\nRunning test review...")
            result = reviewer.review([test_img])
            print(f"Result: {json.dumps(result, indent=2)}")
    else:
        print("Model files not found. Download first:")
        print("  https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF")
        print(f"  Save to: {base}/models/llm/")
