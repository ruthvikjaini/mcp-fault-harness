import pytest

from mcp_fault_harness.protocol import (
    ToolDefinition,
    ToolError,
    make_request,
    parse_request,
)


def test_envelope_roundtrip():
    envelope = make_request("catalog_price", {"item_id": "widget-a"})
    name, arguments = parse_request(envelope)
    assert name == "catalog_price"
    assert arguments == {"item_id": "widget-a"}


def test_parse_rejects_non_object_and_wrong_method():
    with pytest.raises(ToolError):
        parse_request("not a dict")
    with pytest.raises(ToolError):
        parse_request({"jsonrpc": "2.0", "method": "tools/list", "params": {}})


def test_validate_arguments_missing_and_wrong_type():
    definition = ToolDefinition(
        name="t", description="", required=("amount",), properties={"amount": "number"}
    )
    with pytest.raises(ToolError):
        definition.validate_arguments({})
    with pytest.raises(ToolError):
        definition.validate_arguments({"amount": "ten"})
    definition.validate_arguments({"amount": 3})


def test_validate_payload_missing_field_and_stale():
    definition = ToolDefinition(name="t", description="")
    valid, _ = definition.validate_payload({"value": 1, "as_of": 100}, now_ms=100)
    assert valid
    valid, reason = definition.validate_payload({"as_of": 100}, now_ms=100)
    assert not valid and "value" in reason
    valid, reason = definition.validate_payload(
        {"value": 1, "as_of": 0}, now_ms=100_000
    )
    assert not valid and "stale" in reason
