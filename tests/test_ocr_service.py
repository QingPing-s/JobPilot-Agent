from types import SimpleNamespace

import pytest

from src import ocr_service


def test_extract_image_text_returns_lines_and_confidence(monkeypatch):
    monkeypatch.setattr(ocr_service, "_decode_image", lambda _: "decoded-image")

    def engine(image):
        return SimpleNamespace(
            txts=("AAA建材", "人工智能专业", "Python RAG"),
            scores=(0.98, 0.94, 0.88),
        )

    result = ocr_service.extract_image_text(b"image-bytes", "resume.png", engine=engine)

    assert result.text == "AAA建材\n人工智能专业\nPython RAG"
    assert result.lines == ["AAA建材", "人工智能专业", "Python RAG"]
    assert result.average_confidence == pytest.approx(0.9333)


def test_extract_image_text_rejects_unsupported_file():
    with pytest.raises(ocr_service.OCRServiceError, match="暂只支持"):
        ocr_service.extract_image_text(b"content", "resume.pdf")


def test_extract_image_text_rejects_empty_result(monkeypatch):
    monkeypatch.setattr(ocr_service, "_decode_image", lambda _: "decoded-image")

    def engine(image):
        return SimpleNamespace(txts=(), scores=())

    with pytest.raises(ocr_service.OCRServiceError, match="未识别到文字"):
        ocr_service.extract_image_text(b"image-bytes", "resume.jpg", engine=engine)
