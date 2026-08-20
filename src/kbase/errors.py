"""Every failure this service raises on purpose.

`safe_message` is the half of a failure a tenant is allowed to read. A failure
about the caller's own file says so plainly -- "pdf files are not supported yet"
is the entire point of reporting it. A failure about our infrastructure does not:
the provider's message names the embedding host, and that host is the same one
for every tenant on the deployment.
"""

from __future__ import annotations


class KbError(Exception):
    """Base for anything kbase raises deliberately."""

    #: Set when the exception's own message is not fit for a tenant to read.
    public: str = ""

    @property
    def safe_message(self) -> str:
        return self.public or str(self)


class EmbeddingError(KbError):
    """The embedding provider refused, timed out, or answered nonsense."""

    public = "the embedding provider could not be reached or refused the request"


class ExtractError(KbError):
    """A document's bytes could not be turned into text.

    Its message describes the file the caller sent, so it is reported verbatim.
    """


class SchemaError(KbError):
    """The database's shape and the configuration disagree.

    Never a tenant's problem, so its message is for the log and the operator.
    """
