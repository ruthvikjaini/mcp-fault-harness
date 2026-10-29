from mcp_fault_harness.experiment import run_grid
from mcp_fault_harness.scoring import score_run
from mcp_fault_harness.tasks import TASKS, build_tasks


def test_task_expected_values_match_clean_servers():
    from mcp_fault_harness.servers import build_server_set, find_server

    servers = build_server_set()
    for task in build_tasks():
        values = {}
        for step in task.steps:
            arguments = {}
            for name, raw in step.args.items():
                arguments[name] = (
                    values[raw["ref"]] if isinstance(raw, dict) and "ref" in raw else raw
                )
            server = find_server(servers, step.tool)
            result, _ = server.call(step.tool, arguments, now_ms=0)
            assert result.payload["value"] == step.expected, (task.id, step.id)
            values[step.id] = step.expected


def test_grid_reproducible_and_fault_free_resilience_is_one():
    first = run_grid(n_seeds=2)
    second = run_grid(n_seeds=2)
    assert [(s.profile, s.policy, s.capability, s.mean_score) for s in first] == [
        (s.profile, s.policy, s.capability, s.mean_score) for s in second
    ]
    for summary in first:
        if summary.profile == "fault_free":
            assert summary.resilience_ratio == 1.0
        assert 0.0 <= summary.mean_score <= 10.0
        assert 0.0 <= summary.mean_grounded_score <= 10.0


def test_stress_degrades_naive_and_resilient_never_duplicates():
    summaries = run_grid(n_seeds=3)
    by_key = {(s.profile, s.policy, s.capability): s for s in summaries}
    naive_stress = by_key[("stress", "naive", "large")]
    naive_free = by_key[("fault_free", "naive", "large")]
    assert naive_stress.mean_score < naive_free.mean_score
    for key, summary in by_key.items():
        if key[1] == "resilient":
            assert summary.duplicate_write_rate == 0.0


def test_all_tasks_present():
    assert len(TASKS) == 13
    kinds = {t.kind for t in TASKS}
    assert {"live_lookup", "long_chain", "join", "write"} <= kinds
