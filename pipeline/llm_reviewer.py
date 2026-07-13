"""
LLM Fall Review — Qwen3-VL-4B multimodal review for suspected fall events.

Uses llama-cpp-python to run local quantized vision-language model.
Only triggered when classifier outputs "suspected" (threshold_low < P_fall ≤ threshold_high).

Model: Qwen3-VL-2B-Instruct Q4_K_M + mmproj Q8_0
Input:  1-3 keyframe images from the suspected time window
Output: JSON with fall judgment + confidence + reasoning

Usage:
    reviewer = LLMFallReviewer(
        model_path="models/llm/Qwen3-VL-2B-Instruct-Q4_K_M.gguf",
        mmproj_path="models/llm/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
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
    LLAMA_IMPORT_ERROR = None
except Exception as exc:
    HAS_LLAMA = False
    Llama = None
    Qwen25VLChatHandler = None
    LLAMA_IMPORT_ERROR = exc


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


MULTIVIEW_CONTACT_SHEET_PROMPT = """You are an AI safety reviewer for an elderly fall detection system.

You will be shown three contact-sheet images from synchronized surveillance
cameras. Each contact sheet contains 9 ordered frames sampled from an 8-second
window, arranged left-to-right, top-to-bottom as a timeline. Look at all three
views together and observe the person's posture changes across the timeline.

Judge whether the person transitions into a fall-risk / collapsed posture, or
remains clearly in normal daily activity. Prefer safety: if one or more clear
views show a person transitioning to lying, prone, or collapsed on the floor or
a mat, mark it as a fall.

Output your judgment as a JSON object with exactly these fields:
- "is_fall": true or false
- "confidence": a number from 0.0 to 1.0
- "reason": a brief 1-2 sentence explanation in English

IMPORTANT: Only output the JSON object, nothing else.

Example output:
{{"is_fall": true, "confidence": 0.92, "reason": "Across the views, the person transitions from upright to lying on the mat."}}
"""


SINGLE_VIEW_TIMELINE_DESCRIPTION_PROMPT = """You are describing one camera view for an elderly fall detection system.

You will be shown 9 ordered frames from the same camera. The middle frame is
the center of the most suspicious 40-frame 3D classifier window. The four
earlier frames are sampled every 15 frames before it, and the four later frames
are sampled every 15 frames after it. Treat them as a short timeline.

Do not make a final multi-camera decision. Describe only what this camera view
shows: posture, motion trend, whether the person becomes lower/horizontal, and
whether visibility is clear or occluded.

Output valid JSON only with exactly these fields:
- "view": "{view_label}"
- "visibility": "clear", "partial", or "poor"
- "timeline_description": a concise 1-2 sentence description of the motion
- "fall_evidence": a concise 1 sentence description of any fall-like evidence, or "none"
- "view_suggests_fall": true or false
- "confidence": a number from 0.0 to 1.0

IMPORTANT: Only output the JSON object, nothing else.
"""


TEXT_FUSION_PROMPT = """You are the final safety reviewer for an elderly fall detection system.

You are given three text-only observations from synchronized camera views.
These observations were produced from 9 ordered frames per camera around the
same suspicious event. You will not receive images in this step.

Use the three view descriptions together. Prefer safety: if one or more clear
views show a transition into a collapsed, horizontal, prone, supine, or
fall-risk posture, classify the event as a fall. If evidence is unclear or
conflicting, keep the event suspicious instead of calling it normal.

Camera observations:
{view_descriptions}

Output valid JSON only with exactly these fields:
- "is_fall": true or false
- "confidence": a number from 0.0 to 1.0
- "combined_description": a concise 1-2 sentence multi-view summary
- "reason": a concise explanation of the final decision

IMPORTANT: Only output the JSON object, nothing else.
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
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
        n_batch: int = 2048,
        n_ubatch: int = 64,
        max_image_side: int = 0,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: Path to GGUF quantized model.
            mmproj_path: Path to mmproj vision projector file.
            n_ctx: Context window size (keep small for low VRAM).
            n_gpu_layers: GPU layers; -1 = all, reduce if OOM.
            n_batch: Logical batch size for llama.cpp compute buffers.
            n_ubatch: Physical micro-batch size; lower values reduce VRAM spikes.
            verbose: Print llama.cpp logs.
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.max_image_side = max_image_side
        self.ready = False

        if not HAS_LLAMA:
            print(
                "[LLMFallReviewer] llama-cpp-python is unavailable. "
                f"Import error: {LLAMA_IMPORT_ERROR}"
            )
            return

        try:
            if Qwen25VLChatHandler is None:
                raise RuntimeError("Qwen VL chat handler is unavailable in llama-cpp-python")
            if hasattr(llama_cpp, "llama_supports_gpu_offload") and not llama_cpp.llama_supports_gpu_offload():
                print(
                    "[LLMFallReviewer] WARNING: llama-cpp-python was built without GPU offload; "
                    "Qwen3-VL will run on CPU even when n_gpu_layers is set."
                )
            chat_handler = Qwen25VLChatHandler(
                clip_model_path=mmproj_path,
                verbose=verbose,
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
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
        max_tokens: int = 384,
        timeout: float = 30.0,
    ) -> dict:
        """Review keyframes for fall event.

        Args:
            frames: BGR images (H×W×3 numpy arrays).
            timeout: Max seconds for LLM response.

        Returns:
            dict with keys: is_fall, confidence, reason, raw_response.
            On error: is_fall=None, raw_response="ERROR: ..."
        """
        if not self.ready:
            return _error("Model not loaded")

        images_base64 = []
        for i, frame in enumerate(frames):
            frame = resize_for_llm(frame, self.max_image_side)
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
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
            )
            raw = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return _error(f"LLM inference failed: {e}")

        return _parse_response(raw)

    def complete_text_json(self, prompt_text: str, max_tokens: int = 256) -> dict:
        """Run a text-only JSON completion."""
        if not self.ready:
            return _error("Model not loaded")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                top_p=0.9,
            )
            raw = response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return _error(f"LLM text fusion failed: {e}")
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

    def review_multiview_contact_sheets(self, sheets: list[np.ndarray], video_path: str | None = None) -> dict:
        """Review one 9-frame contact sheet per camera view."""
        result = self.review(
            sheets,
            prompt_text=MULTIVIEW_CONTACT_SHEET_PROMPT,
        )
        if video_path is not None:
            result["video_path"] = video_path
        result["sampled_frames"] = 9
        result["contact_sheet"] = True
        result["views"] = len(sheets)
        result["review_mode"] = "direct_multiview_contact_sheets"
        return result

    def describe_single_view_sequence(self, frames: list[np.ndarray], view_label: str) -> dict:
        """Describe one camera's 9 ordered frames without making the final decision."""
        sheet = make_contact_sheet(frames, columns=3, tile_width=128, label_prefix=f"{view_label}_t")
        result = self.review(
            [sheet],
            prompt_text=SINGLE_VIEW_TIMELINE_DESCRIPTION_PROMPT.format(view_label=view_label),
            max_tokens=512,
        )
        result["view"] = view_label
        result["sampled_frames"] = len(frames)
        result["contact_sheet"] = True
        return result

    def review_multiview_text_fusion(
        self,
        view_sequences: list[list[np.ndarray]],
        view_labels: list[str] | None = None,
        video_path: str | None = None,
    ) -> dict:
        """Describe each camera separately, then fuse the text observations."""
        if view_labels is None:
            view_labels = [f"cam{i + 1}" for i in range(len(view_sequences))]

        view_results = []
        for label, frames in zip(view_labels, view_sequences):
            if not frames:
                view_results.append({
                    "view": label,
                    "visibility": "poor",
                    "timeline_description": "No frames were available for this camera.",
                    "fall_evidence": "none",
                    "view_suggests_fall": False,
                    "confidence": 0.0,
                })
                continue
            view_results.append(self.describe_single_view_sequence(frames, label))

        compact_view_results = [compact_view_result(result, label) for result, label in zip(view_results, view_labels)]
        view_descriptions = json.dumps(compact_view_results, ensure_ascii=False, separators=(",", ":"))
        fused = self.complete_text_json(
            TEXT_FUSION_PROMPT.format(view_descriptions=view_descriptions),
            max_tokens=256,
        )
        fused["views"] = len(view_results)
        fused["view_descriptions"] = view_results
        fused["sampled_frames"] = len(view_sequences[0]) if view_sequences else 0
        fused["fusion_mode"] = "single_view_descriptions_then_text_fusion"
        fused["review_mode"] = "single_view_descriptions_then_text_fusion"
        if video_path is not None:
            fused["video_path"] = video_path
        return fused

    def count_text_tokens(self, text: str, add_bos: bool = False) -> int | None:
        """Count tokenizer tokens for text-only prompt budgeting."""
        if not self.ready or self.llm is None:
            return None
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True))
        except TypeError:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos))

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
    # Strip markdown code fences first (common LLM output format)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

    # Try direct JSON parse on cleaned text
    try:
        parsed = json.loads(cleaned)
        return _validate(parsed, cleaned)
    except json.JSONDecodeError:
        pass

    # Try direct parse on original text
    try:
        parsed = json.loads(raw)
        return _validate(parsed, raw)
    except json.JSONDecodeError:
        pass

    # Try to extract any JSON object (finds outermost braces)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
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
    normalized = dict(parsed)
    if "is_fall" in parsed:
        is_fall = bool(parsed.get("is_fall", False))
    elif "view_suggests_fall" in parsed:
        is_fall = bool(parsed.get("view_suggests_fall", False))
    else:
        is_fall = False
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    normalized["is_fall"] = is_fall
    normalized["confidence"] = round(confidence, 3)
    normalized["reason"] = str(
        parsed.get(
            "reason",
            parsed.get(
                "combined_description",
                parsed.get("timeline_description", parsed.get("fall_evidence", "")),
            ),
        )
    )[:300]
    normalized["raw_response"] = raw
    return normalized


def compact_view_result(result: dict, fallback_view: str) -> dict:
    """Keep only fields the text-fusion prompt actually needs."""
    return {
        "view": result.get("view", fallback_view),
        "visibility": result.get("visibility", "poor"),
        "timeline_description": result.get("timeline_description", result.get("reason", "")),
        "fall_evidence": result.get("fall_evidence", "none"),
        "view_suggests_fall": bool(result.get("view_suggests_fall", result.get("is_fall", False))),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
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


def resize_for_llm(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale BGR/RGB image before llama.cpp vision encoding."""
    if max_side <= 0:
        return frame
    import cv2

    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


# --- Quick test ---
if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(base, "models", "llm", "Qwen3-VL-2B-Instruct-Q4_K_M.gguf")
    mmproj = os.path.join(base, "models", "llm", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf")

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
        print("  https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF")
        print(f"  Save to: {base}/models/llm/")
