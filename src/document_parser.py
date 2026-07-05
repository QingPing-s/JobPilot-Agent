from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

from .ocr_service import SUPPORTED_IMAGE_SUFFIXES, OCRServiceError, extract_image_text

SUPPORTED_DOCUMENT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {
    ".docx",
    ".json",
    ".md",
    ".pdf",
    ".txt",
}
DEFAULT_MAX_DOCUMENT_BYTES = 15 * 1024 * 1024
DEFAULT_MAX_PDF_PAGES = 10
DEFAULT_MAX_EXTRACTED_CHARS = 200_000
MIN_NATIVE_PDF_PAGE_CHARS = 30


class DocumentParserError(ValueError):
    """Raised when an uploaded resume document cannot be parsed safely."""


@dataclass(frozen=True)
class DocumentExtraction:
    text: str
    extraction_method: str
    page_count: int = 1
    line_count: int = 0
    average_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _max_document_bytes() -> int:
    configured_mb = max(1, int(os.getenv("JOBPILOT_DOCUMENT_MAX_MB", "15")))
    return configured_mb * 1024 * 1024


def _max_pdf_pages() -> int:
    return max(1, int(os.getenv("JOBPILOT_DOCUMENT_MAX_PAGES", str(DEFAULT_MAX_PDF_PAGES))))


def _max_extracted_chars() -> int:
    return max(
        1,
        int(os.getenv("JOBPILOT_DOCUMENT_MAX_EXTRACTED_CHARS", str(DEFAULT_MAX_EXTRACTED_CHARS))),
    )


def _clean_text(text: str) -> str:
    normalized = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if not normalized:
        raise DocumentParserError("文档中没有可提取的文字。")
    if len(normalized) > _max_extracted_chars():
        raise DocumentParserError("文档提取文本过长，请精简后重新上传。")
    return normalized


def _decode_plain_text(document_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return document_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentParserError("文本编码无法识别，请将文件保存为 UTF-8 后重试。")


def _extract_markdown_text(document_bytes: bytes) -> str:
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError("Markdown 解析依赖未安装，请安装 markdown-it-py。") from exc

    source = _decode_plain_text(document_bytes)
    tokens = MarkdownIt("commonmark").parse(source)
    parts: list[str] = []
    for token in tokens:
        if token.type == "inline":
            inline = []
            for child in token.children or []:
                if child.type in {"text", "code_inline"}:
                    inline.append(child.content)
                elif child.type in {"softbreak", "hardbreak"}:
                    inline.append("\n")
                elif child.type == "image":
                    inline.append(child.content)
            value = "".join(inline).strip()
            if value:
                parts.append(value)
        elif token.type in {"code_block", "fence"} and token.content.strip():
            parts.append(token.content.strip())
    return "\n".join(parts)


def _extract_docx_text(document_bytes: bytes) -> str:
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as exc:
        raise RuntimeError("Word 解析依赖未安装，请安装 python-docx。") from exc

    try:
        document = Document(BytesIO(document_bytes))
    except Exception as exc:
        raise DocumentParserError("无法解析 Word 文档，请确认文件是有效的 DOCX。") from exc

    parts: list[str] = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                row_text = " | ".join(cell for cell in cells if cell)
                if row_text:
                    parts.append(row_text)
        else:
            value = block.text.strip()
            if value:
                parts.append(value)
    return "\n".join(parts)


def _extract_pdf_text(document_bytes: bytes) -> DocumentExtraction:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PDF 解析依赖未安装，请安装 PyMuPDF。") from exc

    try:
        document = pymupdf.open(stream=document_bytes, filetype="pdf")
    except Exception as exc:
        raise DocumentParserError("无法解析 PDF，请确认文件未损坏。") from exc

    try:
        if document.needs_pass:
            raise DocumentParserError("暂不支持加密 PDF，请先移除密码。")
        page_count = len(document)
        if page_count == 0:
            raise DocumentParserError("PDF 不包含页面。")
        if page_count > _max_pdf_pages():
            raise DocumentParserError(f"PDF 不能超过 {_max_pdf_pages()} 页。")

        page_texts: list[str] = []
        ocr_scores: list[float] = []
        ocr_page_count = 0
        for page_number, page in enumerate(document):
            native_text = page.get_text("text", sort=True).strip()
            visible_chars = len(re.sub(r"\s+", "", native_text))
            if visible_chars >= MIN_NATIVE_PDF_PAGE_CHARS:
                page_texts.append(native_text)
                continue

            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            try:
                extraction = extract_image_text(
                    pixmap.tobytes("png"),
                    f"page_{page_number + 1}.png",
                )
            except OCRServiceError as exc:
                raise DocumentParserError(
                    f"PDF 第 {page_number + 1} 页无法提取文字：{exc}"
                ) from exc
            page_texts.append(extraction.text)
            ocr_scores.append(extraction.average_confidence)
            ocr_page_count += 1

        method = "pdf_ocr" if ocr_page_count == page_count else "pdf_native"
        if 0 < ocr_page_count < page_count:
            method = "pdf_hybrid"
        warnings = []
        if ocr_page_count:
            warnings.append(f"PDF 中有 {ocr_page_count} 页未包含可用文本层，已使用 OCR。")
        confidence = round(sum(ocr_scores) / len(ocr_scores), 4) if ocr_scores else 0.0
        text = _clean_text("\n\n".join(page_texts))
        return DocumentExtraction(
            text=text,
            extraction_method=method,
            page_count=page_count,
            line_count=len(text.splitlines()),
            average_confidence=confidence,
            warnings=warnings,
        )
    finally:
        document.close()


def extract_document_text(document_bytes: bytes, filename: str) -> DocumentExtraction:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".doc":
        raise DocumentParserError("暂不支持旧版 DOC，请另存为 DOCX 后上传。")
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        supported = "、".join(sorted(SUPPORTED_DOCUMENT_SUFFIXES))
        raise DocumentParserError(f"不支持该文件类型。支持：{supported}。")
    if not document_bytes:
        raise DocumentParserError("上传的文档为空。")
    if len(document_bytes) > _max_document_bytes():
        limit_mb = max(1, _max_document_bytes() // (1024 * 1024))
        raise DocumentParserError(f"文档不能超过 {limit_mb} MB。")

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        try:
            image = extract_image_text(document_bytes, filename)
        except OCRServiceError as exc:
            raise DocumentParserError(str(exc)) from exc
        return DocumentExtraction(
            text=image.text,
            extraction_method="image_ocr",
            line_count=len(image.lines),
            average_confidence=image.average_confidence,
        )
    if suffix == ".pdf":
        return _extract_pdf_text(document_bytes)
    if suffix == ".docx":
        text = _extract_docx_text(document_bytes)
        method = "docx"
    elif suffix == ".md":
        text = _extract_markdown_text(document_bytes)
        method = "markdown"
    else:
        text = _decode_plain_text(document_bytes)
        method = "json" if suffix == ".json" else "plain_text"

    text = _clean_text(text)
    if method == "json":
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise DocumentParserError("JSON 文件格式无效。") from exc
    return DocumentExtraction(
        text=text,
        extraction_method=method,
        line_count=len(text.splitlines()),
    )
