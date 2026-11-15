"""
WebSocket API for the dashboard.

Clients connect to ws://host:ws_port/?token=<api_token> (browsers cannot set
WebSocket headers, so the token travels in the query string; use it on the
kit's own network only). With an empty api_token, authentication is off.

Server -> client messages carry "schema": SCHEMA_VERSION:
  {"type": "hello", ...}       once on connect: nodes, positions, modes, model
  {"type": "rescue_ai", ...}   snapshot at broadcast_hz (see Pipeline.snapshot)
  {"type": "error", "error": "..."}

  {"type": "setup_state", ...}  USB boards and flashing progress (see setup.py)

Client -> server:
  {"type": "set_op_mode", "mode": 0|1|2}
  {"type": "setup_probe" | "setup_flash", ...}   laptop-local clients only
  {"type": "record_start" | "record_stop" | "label_start" | "label_stop" | "label_add" |
           "label_update" | "label_delete" | "recording_delete" | "data_refresh", ...}
"""

from __future__ import annotations

import hmac
import json
import logging
from collections.abc import Callable
from http import HTTPStatus
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection, broadcast, serve
from websockets.http11 import Request, Response

from .config import OP_MODES, Config
from .pipeline import SCHEMA_VERSION

log = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 4096
DATA_MESSAGES = {"record_start", "record_stop", "label_start", "label_stop", "label_add",
                 "label_update", "label_delete", "recording_delete", "data_refresh"}
LOCAL_HOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}


def _request_type(message: str | bytes) -> Optional[str]:
    """The "type" of a client message, for tagging error replies; None if unreadable."""
    try:
        data = json.loads(message)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    kind = data.get("type") if isinstance(data, dict) else None
    return kind if isinstance(kind, str) else None


def is_local(ws: Optional[ServerConnection]) -> bool:
    addr = getattr(ws, "remote_address", None)
    return addr is not None and len(addr) > 0 and addr[0] in LOCAL_HOSTS


class Api:
    def __init__(self, cfg: Config, hello: Callable[[], dict[str, Any]],
                 on_set_mode: Optional[Callable[[int], None]] = None):
        self.cfg = cfg
        self._hello = hello
        self._on_set_mode = on_set_mode
        self.setup: Any = None          # BoardSetup when running live
        self.data: Any = None           # DataCollection when running live
        self.clients: set[ServerConnection] = set()
        if not cfg.api_token:
            log.warning("api_token is empty: dashboard WebSocket has no authentication")

    async def start(self, host: Optional[str] = None, port: Optional[int] = None) -> Server:
        return await serve(self._handler,
                           host if host is not None else self.cfg.ws_host,
                           port if port is not None else self.cfg.ws_port,
                           process_request=self._authorize, max_size=MAX_MESSAGE_BYTES)

    def _authorize(self, connection: ServerConnection, request: Request) -> Optional[Response]:
        if not self.cfg.api_token:
            return None
        token = parse_qs(urlsplit(request.path).query).get("token", [""])[0]
        if hmac.compare_digest(token.encode(), self.cfg.api_token.encode()):
            return None
        log.warning("Rejected dashboard connection with a bad or missing token")
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

    async def _handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        log.info("Dashboard connected (%d total)", len(self.clients))
        try:
            await ws.send(json.dumps({**self._hello(), "type": "hello", "schema": SCHEMA_VERSION}))
            async for message in ws:
                error = self._handle_message(message, ws)
                if error:
                    await ws.send(json.dumps({"type": "error", "schema": SCHEMA_VERSION,
                                              "error": error, "request": _request_type(message)}))
        finally:
            self.clients.discard(ws)
            log.info("Dashboard disconnected (%d total)", len(self.clients))

    def _handle_message(self, message: str | bytes, ws: Optional[ServerConnection] = None) -> Optional[str]:
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "invalid JSON"
        if not isinstance(data, dict):
            return "message must be a JSON object"
        if data.get("type") == "set_op_mode":
            mode = data.get("mode")
            if not isinstance(mode, int) or isinstance(mode, bool) or mode not in OP_MODES:
                return f"mode must be one of {sorted(OP_MODES)}"
            if self._on_set_mode is None:
                return "mode changes are not available in replay"
            self._on_set_mode(mode)
            log.info("Operating mode set to %d (%s)", mode, OP_MODES[mode])
            return None
        kind = data.get("type")
        if kind in DATA_MESSAGES:
            if self.data is None:
                return "recordings and labels are only available while running live (python -m trifi run)"
            answer: Optional[str] = self.data.handle(data)
            return answer
        if isinstance(kind, str) and kind.startswith("setup_"):
            if self.setup is None:
                return "board setup is only available while running live (python -m trifi run)"
            if not is_local(ws):
                return "board setup only works on the laptop the boards are plugged into"
            result: Optional[str] = self.setup.handle(data)
            return result
        return f"unknown message type {kind!r}"

    def broadcast(self, payload: dict[str, Any]) -> None:
        if self.clients:
            broadcast(self.clients, json.dumps(payload))
