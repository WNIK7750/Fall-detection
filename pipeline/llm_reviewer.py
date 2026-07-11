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

import json
import re
import base64
import io
from typing import Optional

import numpy as np

# Try importing llama-cpp; handle gracefully if not installed
try:
    import llama_cpp
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False
    Llama = None


FALL_REVIEW_PROMPT = """You are an AI safety reviewer for an elderly fall detection system.

You will be shown {n_frames} keyframe images from a multi-camera surveillance video.
Your task: carefully inspect the images and determine whether the person in the scene
is experiencing a fall (collapsed on the ground, fallen from standing position,
lying motionless, struggling to get up after a fall) OR is in normal daily activity.

Output your judgment as a JSON object with exactly these fields:
- "is_fall": true or false (whether a fall event is occurring)
- "confidence": a number from 0.0 to 1.0 indicating your certainty
- "reason": a brief explanation in English, 1-2 sentences

IMPORTANT: Only output the JSON object, nothing else. Do not add any other text.

Example output:
{{"is_fall": true, "confidence": 0.92, "reason": "Person is lying on floor."}}
"""

SYSTEM_PROMPT = "You are a computer vision safety reviewer. Always respond with valid JSON only."


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
            self.llm = Llama(
                model_path=model_path,
                mmproj=mmproj_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                chat_format="chatml",
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
            _, buffer = cv2_imencode(frame_rgb)
            images_base64.append(base64.b64encode(buffer).decode("utf-8"))

        # Build prompt
        prompt_text = FALL_REVIEW_PROMPT.format(n_frames=len(frames))

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
