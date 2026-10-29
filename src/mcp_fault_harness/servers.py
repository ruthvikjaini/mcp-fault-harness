"""Simulated MCP servers: deterministic tools with no network.

Each server exposes tools through ``tools/list``-style definitions and
a ``tools/call``-style dispatch. Data is fixed, so a task's expected
answer can be derived from the same pure functions the servers use
(``price_of``, ``rate_of``, ...). Servers are created fresh for every
run: write effects never leak between runs, which is what makes
duplicate-effect counting meaningful.

Latency is simulated in virtual milliseconds — no real sleeping — so
a full experiment grid runs in seconds and is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_fault_harness.protocol import ToolDefinition, ToolError, ToolResult

ITEMS: dict[str, dict[str, float]] = {
    "widget-a": {"price": 120.0, "stock": 40},
    "widget-b": {"price": 85.0, "stock": 12},
    "gadget-c": {"price": 240.0, "stock": 7},
    "gadget-d": {"price": 60.0, "stock": 0},
}

RATES: dict[str, float] = {
    "USD/EUR": 0.92,
    "USD/GBP": 0.79,
    "EUR/GBP": 0.86,
}

CUSTOMERS: dict[str, dict[str, float]] = {
    "cust-1": {"credit": 5000.0, "balance": 1200.0},
    "cust-2": {"credit": 2500.0, "balance": 2400.0},
    "cust-3": {"credit": 8000.0, "balance": 300.0},
}

USAGE: dict[str, float] = {
    "acct-1": 340.0,
    "acct-2": 1250.0,
    "acct-3": 85.0,
}


def price_of(item_id: str) -> float:
    return ITEMS[item_id]["price"]


def stock_of(item_id: str) -> float:
    return ITEMS[item_id]["stock"]


def rate_of(pair: str) -> float:
    return RATES[pair]


def credit_of(customer_id: str) -> float:
    return CUSTOMERS[customer_id]["credit"]


def usage_of(account_id: str) -> float:
    return USAGE[account_id]


def discount(amount: float, percent: float) -> float:
    return round(amount * (1.0 - percent / 100.0), 4)


def with_tax(amount: float, rate_percent: float) -> float:
    return round(amount * (1.0 + rate_percent / 100.0), 4)


def convert(amount: float, pair: str) -> float:
    return round(amount * RATES[pair], 4)


Handler = Callable[[dict[str, Any]], Any]


@dataclass
class Server:
    """One simulated MCP server hosting a set of tools."""

    name: str
    definitions: dict[str, ToolDefinition] = field(default_factory=dict)
    handlers: dict[str, Handler] = field(default_factory=dict)
    latencies: dict[str, int] = field(default_factory=dict)
    # Write-effect log: every applied write appends one entry here.
    effects: list[dict[str, Any]] = field(default_factory=list)
    # Server-side idempotency store: key -> stored payload.
    idempotency: dict[str, dict[str, Any]] = field(default_factory=dict)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self.definitions.values())

    def call(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        now_ms: int,
        idempotency_key: str | None = None,
    ) -> tuple[ToolResult, bool]:
        """Dispatch a call. Returns ``(result, applied)``.

        ``applied`` is True when a write effect was applied by this
        call. It is how the injector knows a ``timeout_after_apply``
        fault is ambiguous: the effect exists even though the caller
        never sees the response.
        """
        definition = self.definitions.get(tool)
        if definition is None:
            raise ToolError("server_error", f"unknown tool {tool!r}", 10)
        definition.validate_arguments(arguments)
        if idempotency_key is not None and idempotency_key in self.idempotency:
            stored = self.idempotency[idempotency_key]
            return (
                ToolResult(
                    tool=tool,
                    payload=dict(stored),
                    latency_ms=self.latencies.get(tool, 50),
                ),
                False,
            )
        value = self.handlers[tool](arguments)
        latency = self.latencies.get(tool, 50)
        if tool in ("place_order", "update_credit"):
            effect_id = self._effect_id(tool, arguments)
            payload: dict[str, Any] = {
                "effect_id": effect_id,
                "value": value,
                "as_of": now_ms,
            }
            self.effects.append(
                {"tool": tool, "arguments": dict(arguments), "effect_id": effect_id}
            )
            if idempotency_key is not None:
                self.idempotency[idempotency_key] = dict(payload)
            return ToolResult(tool=tool, payload=payload, latency_ms=latency), True
        payload = {"value": value, "as_of": now_ms}
        return ToolResult(tool=tool, payload=payload, latency_ms=latency), False

    @staticmethod
    def _effect_id(tool: str, arguments: dict[str, Any]) -> str:
        parts = "-".join(f"{k}={v}" for k, v in sorted(arguments.items()))
        return f"{tool}:{parts}"


def build_server_set() -> dict[str, Server]:
    """Build the full set of simulated servers, fresh for one run."""

    def need(arguments: dict[str, Any], name: str) -> Any:
        if name not in arguments:
            raise ToolError("malformed", f"missing argument {name!r}", 10)
        return arguments[name]

    catalog = Server(name="catalog")
    catalog.definitions = {
        "catalog_price": ToolDefinition(
            name="catalog_price",
            description="Current price of an item.",
            required=("item_id",),
            properties={"item_id": "string"},
        ),
        "inventory_stock": ToolDefinition(
            name="inventory_stock",
            description="Units in stock for an item.",
            required=("item_id",),
            properties={"item_id": "string"},
        ),
    }
    catalog.handlers = {
        "catalog_price": lambda a: price_of(str(need(a, "item_id"))),
        "inventory_stock": lambda a: stock_of(str(need(a, "item_id"))),
    }
    catalog.latencies = {"catalog_price": 60, "inventory_stock": 60}

    billing = Server(name="billing")
    billing.definitions = {
        "exchange_rate": ToolDefinition(
            name="exchange_rate",
            description="Conversion rate for a currency pair.",
            required=("pair",),
            properties={"pair": "string"},
        ),
        "customer_credit": ToolDefinition(
            name="customer_credit",
            description="Credit limit for a customer.",
            required=("customer_id",),
            properties={"customer_id": "string"},
        ),
        "account_usage": ToolDefinition(
            name="account_usage",
            description="Usage units for an account.",
            required=("account_id",),
            properties={"account_id": "string"},
        ),
        "apply_discount": ToolDefinition(
            name="apply_discount",
            description="Price after a percentage discount.",
            required=("amount", "percent"),
            properties={"amount": "number", "percent": "number"},
        ),
        "add_tax": ToolDefinition(
            name="add_tax",
            description="Amount after tax.",
            required=("amount", "rate_percent"),
            properties={"amount": "number", "rate_percent": "number"},
        ),
        "convert_amount": ToolDefinition(
            name="convert_amount",
            description="Amount converted at the current rate.",
            required=("amount", "pair"),
            properties={"amount": "number", "pair": "string"},
        ),
    }
    billing.handlers = {
        "exchange_rate": lambda a: rate_of(str(need(a, "pair"))),
        "customer_credit": lambda a: credit_of(str(need(a, "customer_id"))),
        "account_usage": lambda a: usage_of(str(need(a, "account_id"))),
        "apply_discount": lambda a: discount(
            float(need(a, "amount")), float(need(a, "percent"))
        ),
        "add_tax": lambda a: with_tax(
            float(need(a, "amount")), float(need(a, "rate_percent"))
        ),
        "convert_amount": lambda a: convert(
            float(need(a, "amount")), str(need(a, "pair"))
        ),
    }
    billing.latencies = {
        "exchange_rate": 120,
        "customer_credit": 90,
        "account_usage": 90,
        "apply_discount": 40,
        "add_tax": 40,
        "convert_amount": 110,
    }

    orders = Server(name="orders")
    orders.definitions = {
        "place_order": ToolDefinition(
            name="place_order",
            description="Place an order. Has a side effect; use an idempotency key.",
            required=("item_id", "quantity"),
            properties={"item_id": "string", "quantity": "number"},
            read_only=False,
            required_payload=("effect_id", "value", "as_of"),
        ),
        "update_credit": ToolDefinition(
            name="update_credit",
            description="Set a customer's credit limit. Has a side effect.",
            required=("customer_id", "new_limit"),
            properties={"customer_id": "string", "new_limit": "number"},
            read_only=False,
            required_payload=("effect_id", "value", "as_of"),
        ),
    }
    orders.handlers = {
        "place_order": lambda a: round(
            price_of(str(need(a, "item_id"))) * float(need(a, "quantity")), 4
        ),
        "update_credit": lambda a: float(need(a, "new_limit")),
    }
    orders.latencies = {"place_order": 180, "update_credit": 150}

    servers = {s.name: s for s in (catalog, billing, orders)}
    return servers


def all_definitions(servers: dict[str, Server]) -> dict[str, ToolDefinition]:
    definitions: dict[str, ToolDefinition] = {}
    for server in servers.values():
        definitions.update(server.definitions)
    return definitions


def find_server(servers: dict[str, Server], tool: str) -> Server:
    for server in servers.values():
        if tool in server.definitions:
            return server
    raise ToolError("server_error", f"no server hosts tool {tool!r}", 10)
