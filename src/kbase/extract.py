"""Bytes -> text. P1 handles what decodes; PDF and DOCX arrive in P2."""

from __future__ import annotations

from kbase.errors import ExtractError

_TEXT_SUFFIXES = (".txt", ".md", ".markdown", "")
_KNOWN_BINARY = {".pdf": "pdf", ".docx": "docx", ".doc": "doc"}


def _suffix(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[1].lower()


def extract_text(data: bytes, *, filename: str = "", mime: str = "") -> str:
    suffix = _suffix(filename)
    if suffix in _KNOWN_BINARY:
        raise ExtractError(f"{_KNOWN_BINARY[suffix]} files are not supported yet (planned for P2)")
    if suffix and suffix not in _TEXT_SUFFIXES and not mime.startswith("text/"):
        raise ExtractError(f"unsupported file type: {suffix}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExtractError("file is not valid UTF-8 text") from exc
