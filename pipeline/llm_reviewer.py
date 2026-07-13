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


FALL_REVIEW_PROMPT = """你是老年人跌倒检测系统的视觉复核员。

你会看到{n_frames}张按时间顺序采样的画面。请只根据画面中的动作判断，不要因为这是疑似事件就默认摔倒。

判定标准：
- true：明显失衡、快速失控下降、撞击/倒地，或倒下后异常横躺/趴卧/仰卧且未及时恢复。
- false：蹲下、弯腰/捡东西、缓慢坐下或躺下、扶着支撑物下降、从坐姿/躺姿起身。
- 只看到身体变低、坐着、躺着、靠近沙发/垫子，不足以判定摔倒；重点看动作方向、速度和是否可控。
- MCFD中落在软垫或沙发上不排除摔倒；若动作明显失控仍判true。
- 证据不清时判false并降低confidence，交给人工复核。

只输出JSON：{{"is_fall": true或false, "confidence": 0.0到1.0, "reason": "中文简短原因"}}
"""


CAMERA_CONTACT_SHEET_PROMPT = """你是老年人居家跌倒检测系统的视觉复核员。

你会看到同一个摄像头（{view_label}）的8张四宫格图片。
每张四宫格是2x2时间序列（从左到右、从上到下），8张图连续覆盖围绕可疑动作中心采样出的32帧。
每个小图左上角标有全局时间编号t00到t31；t00最早，t31最晚。必须按t00 -> t31判断，不能倒着看。
这些画面来自3D分类器筛出的疑似窗口。

画面说明：
- 镜头存在一定畸变，画面边缘的人体和地面线条可能会变形。
- 屏幕中间区域是底板，底板上贴有网格胶带，能看到银色和红色胶带。
- 人正常站立时，身体理论上接近竖直方向，头部在上方。
- 判断动作时优先看人体相对底板网格的姿态变化和运动连续性，不要只凭边缘畸变后的形状下结论。

请像人工复核员一样先理解这段动作，而不是只看最后一帧。可以按这些线索思考：
- 起点和终点：人一开始大致是站立、坐着、蹲着还是躺着？最后是更低、更高，还是姿态稳定？
- 重心路径：身体是在主动、有节奏地下移/上移，还是突然失去平衡后被动下坠？
- 支撑方式：脚、膝、手、沙发/床沿是否在持续支撑身体？动作看起来是否有目的？
- 接触方式：接触地面/垫子/沙发时像正常坐卧，还是像撞击、滑倒、塌落？
- 恢复趋势：后续画面是否有坐稳、站起、调整姿势等主动恢复？

请在以下动作类别中选择最贴近的一类：
- "fall"：失衡、滑倒、塌落、被动下坠，最后形成异常倒地姿态。
- "squat_bend"：蹲下、深蹲、弯腰、捡东西，通常有膝/脚/手的主动支撑。
- "sit_down"：坐到椅子、沙发、床或垫子上，下降过程较可控。
- "lie_down"：主动躺下或调整到躺姿。
- "stand_up"：从坐姿、躺姿或蹲姿起身。
- "unclear"：遮挡、画面不足，或几个类别难以区分。

is_fall 应与 action 保持一致：只有 action 为 "fall" 时才输出 true。reason 用一句话说明最关键的视觉线索即可。

只输出JSON：{{"is_fall": true或false, "confidence": 0.0到1.0, "action": "fall/squat_bend/sit_down/lie_down/stand_up/unclear", "reason": "中文简短原因"}}
"""

MULTIVIEW_CONTACT_SHEET_PROMPT = CAMERA_CONTACT_SHEET_PROMPT.format(view_label="camera")

SYSTEM_PROMPT = "你是一个视觉安全审核员。只输出合法的JSON，不要markdown或其他文字。"

RETRY_FORMAT_PROMPT_1 = """上一次的回答无法解析为合法JSON。
请重新看图。基于之前的分析，只输出JSON对象，不要markdown或其他文字：
字段：is_fall, confidence, reason

之前的分析：
{previous_output}
"""

RETRY_FORMAT_PROMPT_2 = """两次都未能生成合法JSON。综合以下分析，严格输出JSON：
字段：is_fall, confidence, reason
不要markdown，不要其他文字。

第一次分析：
{output_1}

第二次分析：
{output_2}
"""


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
        """Review ordered frames with the unified Chinese fall prompt."""
        return self.review(
            frames,
            prompt_text=FALL_REVIEW_PROMPT.format(n_frames=len(frames)),
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
                prompt_text=FALL_REVIEW_PROMPT.format(n_frames=len(frames)),
            )
        else:
            result = self.review_raw(frames)
        result["video_path"] = video_path
        result["sampled_frames"] = len(frames)
        result["frame_stride"] = frame_stride
        result["contact_sheet"] = contact_sheet
        return result

    def _is_valid_fall_result(self, result: dict) -> bool:
        """Check whether the parsed result contains a clear fall judgment."""
        if result.get("is_fall") is None:
            return False
        reason = str(result.get("reason", "")).strip()
        if not reason or reason.startswith("ERROR:"):
            return False
        return True

    def review_camera_contact_sheets(
        self,
        sheets: list[np.ndarray],
        view_label: str,
        video_path: str | None = None,
        sampled_frames: int | None = None,
    ) -> dict:
        """Review one camera's ordered contact sheets with parse-format retries."""
        prompt = CAMERA_CONTACT_SHEET_PROMPT.format(view_label=view_label)
        result = self.review(
            sheets,
            prompt_text=prompt,
        )
        if video_path is not None:
            result["video_path"] = video_path
        result["sampled_frames"] = sampled_frames if sampled_frames is not None else len(sheets) * 9
        result["contact_sheet"] = True
        result["view"] = view_label
        result["sheets"] = len(sheets)
        result["review_mode"] = "single_camera_81_frame_contact_sheets"
        result["retry_count"] = 0

        if self._is_valid_fall_result(result):
            return result

        # --- Retry 1: with images + previous output + format enforcement ---
        raw_1 = result.get("raw_response", "")
        retry_prompt = prompt + "\n\n" + RETRY_FORMAT_PROMPT_1.format(previous_output=raw_1)
        retry_1 = self.review(
            sheets,
            prompt_text=retry_prompt,
        )
        retry_1["raw_response"] = retry_1.get("raw_response", "")
        if self._is_valid_fall_result(retry_1):
            retry_1["retry_count"] = 1
            retry_1["review_mode"] = "single_camera_retry_1"
            if video_path is not None:
                retry_1["video_path"] = video_path
            retry_1["view"] = view_label
            retry_1["sheets"] = len(sheets)
            retry_1["sampled_frames"] = sampled_frames if sampled_frames is not None else len(sheets) * 9
            retry_1["contact_sheet"] = True
            return retry_1

        # --- Retry 2: final forced format, use both previous outputs ---
        raw_2 = retry_1.get("raw_response", "")
        retry_2 = self.complete_text_json(
            RETRY_FORMAT_PROMPT_2.format(output_1=raw_1, output_2=raw_2),
            max_tokens=192,
        )
        retry_2["raw_response"] = retry_2.get("raw_response", "")
        if self._is_valid_fall_result(retry_2):
            retry_2["retry_count"] = 2
            retry_2["review_mode"] = "single_camera_retry_2"
            if video_path is not None:
                retry_2["video_path"] = video_path
            retry_2["view"] = view_label
            retry_2["sheets"] = len(sheets)
            retry_2["sampled_frames"] = sampled_frames if sampled_frames is not None else len(sheets) * 9
            retry_2["contact_sheet"] = True
            return retry_2

        # --- All 3 attempts failed: escalate to fall ---
        return {
            "is_fall": None,
            "confidence": 0.0,
            "reason": "LLM failed to produce a valid JSON response for this camera.",
            "raw_response": f"RAW_1: {raw_1[:500]}\n---\nRAW_2: {raw_2[:500]}",
            "retry_count": 3,
            "review_mode": "single_camera_invalid",
            "view": view_label,
            "sheets": len(sheets),
            "sampled_frames": sampled_frames if sampled_frames is not None else len(sheets) * 9,
            "contact_sheet": True,
            "video_path": video_path,
        }

    def review_multiview_contact_sheets(self, sheets: list[np.ndarray], video_path: str | None = None) -> dict:
        """Compatibility wrapper for older callers."""
        return self.review_camera_contact_sheets(
            sheets,
            view_label="camera",
            video_path=video_path,
            sampled_frames=len(sheets) * 9,
        )

    def count_text_tokens(self, text: str, add_bos: bool = False) -> int | None:
        """Count tokenizer tokens for text-only prompt budgeting."""
        if not self.ready or self.llm is None:
            return None
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=True))
        except TypeError:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos))

    def review_with_context(self, frames: list[np.ndarray], evidence: str) -> dict:
        """Compatibility wrapper; review with the unified Chinese fall prompt."""
        return self.review(
            frames,
            prompt_text=FALL_REVIEW_PROMPT.format(n_frames=len(frames)),
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
    start_index: int = 0,
    center_crop_fraction: float | None = None,
    center_crop_square: bool = False,
) -> np.ndarray:
    """Pack ordered frames into one labeled timeline image."""
    import cv2

    if not frames:
        raise ValueError("frames must not be empty")
    columns = max(1, columns)
    rows = int(np.ceil(len(frames) / columns))

    h0, w0 = frames[0].shape[:2]
    if center_crop_square or center_crop_fraction is not None:
        tile_height = tile_width
    else:
        tile_height = max(1, int(round(tile_width * h0 / max(w0, 1))))
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        row, col = divmod(i, columns)
        if center_crop_square:
            h, w = frame.shape[:2]
            side = max(1, min(h, w))
            y0_crop = max(0, (h - side) // 2)
            x0_crop = max(0, (w - side) // 2)
            frame = frame[y0_crop:y0_crop + side, x0_crop:x0_crop + side]
        elif center_crop_fraction is not None and 0 < center_crop_fraction <= 1:
            h, w = frame.shape[:2]
            side = max(1, int(round(min(h, w) * center_crop_fraction)))
            y0_crop = max(0, (h - side) // 2)
            x0_crop = max(0, (w - side) // 2)
            frame = frame[y0_crop:y0_crop + side, x0_crop:x0_crop + side]
        tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (72, 24), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{label_prefix}{start_index + i:02d}",
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
