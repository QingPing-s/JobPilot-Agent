from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 20_000_000
_INFERENCE_LOCK = threading.Lock()


class OCRServiceError(ValueError):
    """Raised when an uploaded image cannot be processed safely."""


@dataclass(frozen=True)
class OCRExtraction:
    text: str
    lines: list[str]
    scores: list[float]
    average_confidence: float


def _max_image_bytes() -> int:
    return max(1, int(os.getenv("JOBPILOT_OCR_MAX_IMAGE_MB", "8"))) * 1024 * 1024


def _max_image_pixels() -> int:
    return max(1, int(os.getenv("JOBPILOT_OCR_MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS))))


def _decode_image(image_bytes: bytes) -> Any:
    try:
        import numpy as np
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise RuntimeError("OCR 图片解码依赖未安装，请安装 rapidocr。") from exc

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > _max_image_pixels():
                raise OCRServiceError("图片尺寸过大，请压缩后重试。")
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise OCRServiceError("无法解析图片，请上传有效的 PNG、JPG、WEBP 或 BMP 文件。") from exc


@lru_cache(maxsize=1)
def _get_engine() -> Any:
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("RapidOCR 未安装，请执行 pip install rapidocr onnxruntime。") from exc
    return RapidOCR()


def _to_float_list(values: Any, expected_length: int) -> list[float]:
    if values is None:
        return [0.0] * expected_length
    scores: list[float] = []
    for value in values:
        try:
            scores.append(max(0.0, min(1.0, float(value))))
        except (TypeError, ValueError):
            scores.append(0.0)
    if len(scores) < expected_length:
        scores.extend([0.0] * (expected_length - len(scores)))
    return scores[:expected_length]


def extract_image_text(
    image_bytes: bytes,
    filename: str,
    *,
    engine: Any | None = None,
) -> OCRExtraction:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise OCRServiceError("暂只支持 PNG、JPG、JPEG、WEBP 和 BMP 图片。")
    if not image_bytes:
        raise OCRServiceError("上传的图片为空。")
    if len(image_bytes) > _max_image_bytes():
        raise OCRServiceError(
            f"图片不能超过 {max(1, _max_image_bytes() // (1024 * 1024))} MB。"
        )

    image = _decode_image(image_bytes)
    ocr_engine = engine or _get_engine()
    with _INFERENCE_LOCK:
        result = ocr_engine(image)

    raw_lines = getattr(result, "txts", None)
    if raw_lines is None:
        raise OCRServiceError("RapidOCR 未返回可识别文本。")
    lines = [str(line).strip() for line in raw_lines if str(line).strip()]
    if not lines:
        raise OCRServiceError("图片中未识别到文字，请上传更清晰的简历图片。")

    scores = _to_float_list(getattr(result, "scores", None), len(lines))
    average_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0
    return OCRExtraction(
        text="\n".join(lines),
        lines=lines,
        scores=scores,
        average_confidence=average_confidence,
    )
