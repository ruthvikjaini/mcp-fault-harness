"""Fault-tolerant tool-calling evaluation for MCP-style agents.

Public surface: the fault model (:mod:`mcp_fault_harness.faults`), the
protocol envelope (:mod:`mcp_fault_harness.protocol`), the simulated
servers (:mod:`mcp_fault_harness.servers`), the resilience mechanisms
(:mod:`mcp_fault_harness.resilience`), the simulated agents
(:mod:`mcp_fault_harness.agents`), the task benchmark
(:mod:`mcp_fault_harness.tasks`), scoring
(:mod:`mcp_fault_harness.scoring`), and the experiment grid
(:mod:`mcp_fault_harness.experiment`).

Everything runs on the standard library only. No LLM calls, no network,
no GPU. Faults, servers, and agents are simulated so that resilience
policies can be compared reproducibly; nothing here measures a real
model. See ``docs/RESEARCH.md`` for what comes from the literature and
what is simulation.
"""

from mcp_fault_harness.faults import FaultMode, FaultProfile, FaultSchedule, Persistence
from mcp_fault_harness.protocol import ToolError, ToolResult

__all__ = [
    "FaultMode",
    "FaultProfile",
    "FaultSchedule",
    "Persistence",
    "ToolError",
    "ToolResult",
]
