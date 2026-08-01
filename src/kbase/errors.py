"""Every failure this service raises on purpose."""

from __future__ import annotations


class KbError(Exception):
    """Base for anything kbase raises deliberately."""


class EmbeddingError(KbError):
    """The embedding provider refused, timed out, or answered nonsense."""


class ExtractError(KbError):
    """A document's bytes could not be turned into text."""
