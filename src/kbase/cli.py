"""`kb doctor` and `kb serve`.

`serve` runs `doctor` first and refuses on a failure, because a service that
boots with no keys configured answers 401 to everything while still reporting
healthy to whatever is watching the port.
"""

from __future__ import annotations

import argparse
import os
import sys

from kbase.settings import Settings


def _doctor(settings: Settings) -> int:
    problems = settings.check()
    if not problems:
        print("ok: configuration is complete")
        return 0
    for p in problems:
        print(f"problem: {p}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kb")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check the configuration and exit")
    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8090)

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 2)

    settings = Settings.from_env(os.environ)
    if args.command == "doctor":
        return _doctor(settings)

    if settings.check():
        print("refusing to start:")
        _doctor(settings)
        return 1

    import uvicorn

    from kbase.server.app import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
