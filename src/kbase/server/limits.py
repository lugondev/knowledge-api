"""The one place a request body's size can be refused before it is kept.

A route can only check a size it has already read. `request.json()` and
`request.form()` both drain the whole body into memory first, and `Content-Length`
is absent under chunked transfer -- so on that path the limit was applied to
bytes the process was already holding, and a client that declared nothing could
make it hold as many as it liked. Counting in the ASGI receive channel is
upstream of every reader, so the refusal lands on the chunk that crosses the
line rather than after the last one.
"""

from __future__ import annotations

from fastapi import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLarge(HTTPException):
    """Raised out of `receive`, once the body has outgrown the limit.

    An `HTTPException` specifically. FastAPI reads a declared body inside a
    `try`, and turns anything that escapes into `400 There was an error parsing
    the body` -- except an HTTPException, which it re-raises untouched for
    exactly this case. A plain exception here reached the caller as a 400 about
    malformed JSON, which is neither the right status nor a true description of
    what happened.
    """

    def __init__(self, limit: int) -> None:
        super().__init__(
            status_code=413,
            detail=f"document exceeds KB_MAX_UPLOAD_BYTES ({limit})",
        )
        self.limit = limit


class MaxBodySize:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        received = 0
        limit = self._max_bytes

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        await self._app(scope, counted, send)
