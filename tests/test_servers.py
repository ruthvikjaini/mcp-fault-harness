import pytest

from mcp_fault_harness.protocol import ToolError
from mcp_fault_harness.servers import build_server_set, find_server


def test_clean_read_matches_pure_function():
    servers = build_server_set()
    server = find_server(servers, "catalog_price")
    result, applied = server.call("catalog_price", {"item_id": "widget-a"}, now_ms=10)
    assert result.payload["value"] == 120.0
    assert result.payload["as_of"] == 10
    assert not applied


def test_unknown_tool_and_bad_arguments_raise():
    servers = build_server_set()
    server = find_server(servers, "catalog_price")
    with pytest.raises(ToolError):
        server.call("catalog_price", {}, now_ms=0)
    with pytest.raises(ToolError):
        find_server(servers, "no_such_tool")


def test_write_applies_effect_once_per_call_without_key():
    servers = build_server_set()
    server = find_server(servers, "place_order")
    args = {"item_id": "widget-b", "quantity": 3}
    server.call("place_order", args, now_ms=0)
    server.call("place_order", args, now_ms=0)
    assert len(server.effects) == 2


def test_idempotency_key_dedupes_server_side():
    servers = build_server_set()
    server = find_server(servers, "place_order")
    args = {"item_id": "widget-b", "quantity": 3}
    first, applied_first = server.call(
        "place_order", args, now_ms=0, idempotency_key="k1"
    )
    second, applied_second = server.call(
        "place_order", args, now_ms=0, idempotency_key="k1"
    )
    assert applied_first and not applied_second
    assert first.payload == second.payload
    assert len(server.effects) == 1


def test_compute_tools_chain_from_pure_functions():
    servers = build_server_set()
    billing = find_server(servers, "apply_discount")
    result, _ = billing.call(
        "apply_discount", {"amount": 85.0, "percent": 10}, now_ms=0
    )
    assert result.payload["value"] == 76.5
