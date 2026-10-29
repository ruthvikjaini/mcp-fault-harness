"""The benchmark: twelve tasks across five shapes.

Shapes follow the pattern reported for MCP-Injector — short
computational tasks suffer least, live lookups and long sequential
chains suffer most — but the tasks themselves are this project's own
constructions over the simulated servers in :mod:`servers`. Every
expected value is computed from the same pure functions the servers
use, so ground truth cannot drift from the data.

A step argument may be a literal or ``{"ref": "<step id>"}``, meaning
"use that step's observed value". A chain step therefore inherits
whatever the previous step actually produced — including a guessed or
stale value — which is how one bad tool result cascades into the rest
of a chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_fault_harness.servers import (
    convert,
    credit_of,
    discount,
    price_of,
    rate_of,
    stock_of,
    usage_of,
    with_tax,
)


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    args: dict[str, Any]
    expected: Any
    is_write: bool = False


@dataclass(frozen=True)
class Task:
    id: str
    kind: str  # short_compute | live_lookup | long_chain | join | write
    description: str
    steps: tuple[Step, ...] = field(default_factory=tuple)


def _ref(step_id: str) -> dict[str, Any]:
    return {"ref": step_id}


def build_tasks() -> list[Task]:
    p_a = price_of("widget-a")
    p_b = price_of("widget-b")
    p_c = price_of("gadget-c")
    disc_b = discount(p_b, 10)
    taxed_b = with_tax(disc_b, 8)
    conv_c = convert(p_c, "USD/EUR")
    disc_c = discount(conv_c, 5)
    rate_gbp = rate_of("USD/GBP")
    conv_gbp = convert(1000.0, "USD/GBP")
    disc_gbp = discount(conv_gbp, 10)
    taxed_gbp = with_tax(disc_gbp, 20)
    usage_adj = discount(usage_of("acct-1"), 10)

    return [
        Task(
            id="price_lookup",
            kind="live_lookup",
            description="Look up the current price of widget-a.",
            steps=(Step("s1", "catalog_price", {"item_id": "widget-a"}, p_a),),
        ),
        Task(
            id="stock_check",
            kind="live_lookup",
            description="Check stock for gadget-c.",
            steps=(Step("s1", "inventory_stock", {"item_id": "gadget-c"}, stock_of("gadget-c")),),
        ),
        Task(
            id="fx_lookup",
            kind="live_lookup",
            description="Get the USD/EUR exchange rate.",
            steps=(Step("s1", "exchange_rate", {"pair": "USD/EUR"}, rate_of("USD/EUR")),),
        ),
        Task(
            id="credit_check",
            kind="live_lookup",
            description="Get the credit limit for cust-2.",
            steps=(Step("s1", "customer_credit", {"customer_id": "cust-2"}, credit_of("cust-2")),),
        ),
        Task(
            id="usage_check",
            kind="short_compute",
            description="Get usage for acct-2.",
            steps=(Step("s1", "account_usage", {"account_id": "acct-2"}, usage_of("acct-2")),),
        ),
        Task(
            id="discount_chain",
            kind="long_chain",
            description="Price widget-b, apply a 10% discount, then add 8% tax.",
            steps=(
                Step("s1", "catalog_price", {"item_id": "widget-b"}, p_b),
                Step("s2", "apply_discount", {"amount": _ref("s1"), "percent": 10}, disc_b),
                Step("s3", "add_tax", {"amount": _ref("s2"), "rate_percent": 8}, taxed_b),
            ),
        ),
        Task(
            id="convert_chain",
            kind="long_chain",
            description="Price gadget-c, convert to EUR, then apply a 5% discount.",
            steps=(
                Step("s1", "catalog_price", {"item_id": "gadget-c"}, p_c),
                Step("s2", "convert_amount", {"amount": _ref("s1"), "pair": "USD/EUR"}, conv_c),
                Step("s3", "apply_discount", {"amount": _ref("s2"), "percent": 5}, disc_c),
            ),
        ),
        Task(
            id="join_order_value",
            kind="join",
            description="Price widget-a and check its stock, then discount the price by 20%.",
            steps=(
                Step("s1", "catalog_price", {"item_id": "widget-a"}, p_a),
                Step("s2", "inventory_stock", {"item_id": "widget-a"}, stock_of("widget-a")),
                Step("s3", "apply_discount", {"amount": _ref("s1"), "percent": 20}, discount(p_a, 20)),
            ),
        ),
        Task(
            id="credit_usage_join",
            kind="join",
            description="Get cust-1's credit limit and acct-1's usage, then discount the usage by 10%.",
            steps=(
                Step("s1", "customer_credit", {"customer_id": "cust-1"}, credit_of("cust-1")),
                Step("s2", "account_usage", {"account_id": "acct-1"}, usage_of("acct-1")),
                Step("s3", "apply_discount", {"amount": _ref("s2"), "percent": 10}, usage_adj),
            ),
        ),
        Task(
            id="place_order",
            kind="write",
            description="Place an order for 3 units of widget-b.",
            steps=(
                Step(
                    "s1",
                    "place_order",
                    {"item_id": "widget-b", "quantity": 3},
                    round(p_b * 3, 4),
                    is_write=True,
                ),
            ),
        ),
        Task(
            id="update_credit",
            kind="write",
            description="Raise cust-3's credit limit to 9000.",
            steps=(
                Step(
                    "s1",
                    "update_credit",
                    {"customer_id": "cust-3", "new_limit": 9000},
                    9000.0,
                    is_write=True,
                ),
            ),
        ),
        Task(
            id="fx_chain_4",
            kind="long_chain",
            description="Get USD/GBP, convert 1000 USD, discount 10%, add 20% tax.",
            steps=(
                Step("s1", "exchange_rate", {"pair": "USD/GBP"}, rate_gbp),
                Step("s2", "convert_amount", {"amount": 1000.0, "pair": "USD/GBP"}, conv_gbp),
                Step("s3", "apply_discount", {"amount": _ref("s2"), "percent": 10}, disc_gbp),
                Step("s4", "add_tax", {"amount": _ref("s3"), "rate_percent": 20}, taxed_gbp),
            ),
        ),
        Task(
            id="double_lookup",
            kind="join",
            description="Price widget-a and gadget-c (same tool twice), then discount the first price by 20%. The repeated tool is what lets a circuit breaker engage within one run.",
            steps=(
                Step("s1", "catalog_price", {"item_id": "widget-a"}, p_a),
                Step("s2", "catalog_price", {"item_id": "gadget-c"}, p_c),
                Step("s3", "apply_discount", {"amount": _ref("s1"), "percent": 20}, discount(p_a, 20)),
            ),
        ),
    ]


TASKS: list[Task] = build_tasks()

KINDS: tuple[str, ...] = ("short_compute", "live_lookup", "long_chain", "join", "write")
