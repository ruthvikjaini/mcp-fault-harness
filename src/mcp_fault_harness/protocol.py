"""The MCP-shaped tool-call envelope and its validation.

A call is a JSON-RPC-style request dict with ``jsonrpc``, ``method``
(``tools/call``), ``params.name`` and ``params.arguments``. A result is
either a structured ``content`` payload or a :class:`ToolError`.

Validation is the boundary that separates a crash from silent damage:
an explicit error is visible to every policy, but a structurally wrong
payload (missing field, wrong type, stale timestamp) only fails when
the caller checks the schema and the freshness window. Policies that
skip validation accept bad payloads as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Error kinds. ``retryable`` in faults.py decides which of these a
# retry policy will retry; the kind itself only describes the failure.
ERROR_KINDS = (
    "timeout",
    "timeout_after_apply",
    "transport_error",
    "server_error",
    "rate_limited",
    "auth_expired",
    "malformed",
    "breaker_open",
)


@dataclass(frozen=True)
class ToolError(Exception):
    """An explicit, visible tool failure."""

    kind: str
    message: str
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ERROR_KINDS:
            raise ValueError(f"unknown error kind: {self.kind!r}")

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.kind}: {self.message}"


@dataclass
class ToolResult:
    """A tool response that arrived. The payload may still be bad."""

    tool: str
    payload: dict[str, Any]
    latency_ms: int = 0
    stale: bool = False
    source: str = "live"  # "live", "fallback", or "cache"

    def field(self, name: str) -> Any:
        return self.payload.get(name)


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as advertised by ``tools/list``."""

    name: str
    description: str
    required: tuple[str, ...] = ()
    properties: dict[str, str] = field(default_factory=dict)
    # Freshness window (virtual ms) for payloads carrying ``as_of``.
    freshness_ms: int = 10_000
    read_only: bool = True
    # Payload fields a usable result must carry.
    required_payload: tuple[str, ...] = ("value", "as_of")

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        missing = [name for name in self.required if name not in arguments]
        if missing:
            raise ToolError(
                "malformed", f"missing required arguments: {', '.join(missing)}"
            )
        for name, expected in self.properties.items():
            if name in arguments and expected == "number" and not isinstance(
                arguments[name], (int, float)
            ):
                raise ToolError(
                    "malformed", f"argument {name!r} must be a number"
                )
            if name in arguments and expected == "string" and not isinstance(
                arguments[name], str
            ):
                raise ToolError(
                    "malformed", f"argument {name!r} must be a string"
                )

    def validate_payload(
        self, payload: dict[str, Any], *, now_ms: int
    ) -> tuple[bool, str]:
        """Check a payload against the tool schema and freshness window.

        Returns ``(True, "")`` when the payload is usable evidence, else
        ``(False, reason)``. This is deliberately strict: a missing field
        is not evidence, and neither is a value older than the window.
        """
        for name in self.required_payload:
            if name not in payload:
                return False, f"missing field {name!r}"
        as_of = payload.get("as_of")
        if as_of is not None:
            if not isinstance(as_of, (int, float)):
                return False, "as_of is not a timestamp"
            if now_ms - as_of > self.freshness_ms:
                return False, "payload is stale"
        return True, ""



def make_request(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build a ``tools/call`` request envelope."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": dict(arguments)},
    }


def parse_request(envelope: Any) -> tuple[str, dict[str, Any]]:
    """Parse a request envelope, raising :class:`ToolError` if malformed."""
    if not isinstance(envelope, dict):
        raise ToolError("malformed", "request envelope is not an object")
    if envelope.get("jsonrpc") != "2.0" or envelope.get("method") != "tools/call":
        raise ToolError("malformed", "request is not a tools/call envelope")
    params = envelope.get("params")
    if not isinstance(params, dict):
        raise ToolError("malformed", "request params missing")
    name = params.get("name")
    arguments = params.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise ToolError("malformed", "request name/arguments malformed")
    return name, arguments
