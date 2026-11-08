"""Read the CDL journey using only Python's standard library and JSON files.

No Julia runtime, ExperimentLab API, tensor or JLD2 payload is loaded. This is
an executable frontend contract check, not another numerical implementation.
"""
import json
from pathlib import Path
import sys


def read_json(path):
    with path.open() as stream:
        return json.load(stream)


def inspect_store(root):
    index = read_json(root / "index.json")
    results = []
    for execution_id, execution in index["executions"].items():
        if not execution["label"].startswith(("als", "nnr_")):
            continue
        assert execution["status"] == "completed"
        algorithm_id = execution["parent"]["id"]
        algorithm = index["algorithms"][algorithm_id]
        with (root / execution["log_path"]).open() as stream:
            logs = [json.loads(line) for line in stream if line.strip()]
        steps = [row for row in logs if row.get("event") == "mps.split.update.v1"]
        offset = 400 if execution["label"] == "nnr_resumed" else 0
        assert [row["data"]["completed_updates"] for row in steps] == list(range(offset + 1, 801))
        for row in steps:
            assert row["owner"] == {"kind": "execution", "id": execution_id}
            assert row["store_id"] == index["id"]
            data = row["data"]
            assert data["site"] == (data["completed_updates"] - 1) % 8 + 1
            assert data["sweep"] == (data["completed_updates"] - 1) // 8 + 1
            assert data["duration"] >= 0
            assert data["cost_scale"] > 0
        measurements = []
        for attempt_id, attempt in index["measurements"].items():
            state = index["states"][attempt["parent"]["id"]]
            if state["parent"]["id"] != execution_id:
                continue
            assert attempt["status"] == "completed"
            payload = read_json(root / attempt["result"])
            assert payload["attempt"] == attempt_id
            # Measurement logs have their own owner and are discoverable by path.
            with (root / attempt["log_path"]).open() as stream:
                measurement_logs = [json.loads(line) for line in stream if line.strip()]
            assert measurement_logs
            assert all(row["owner"] == {"kind": "measurement", "id": attempt_id}
                       for row in measurement_logs)
            measurements.append({"step": state["step"], "state_id": state["id"],
                "measurement_id": attempt_id, "rank_rtol": attempt["config"]["parameters"]["rank_rtol"],
                "result_path": attempt["result"], "data": payload["data"]})
        measurements.sort(key=lambda item: item["step"])
        final = measurements[-1]["data"]
        results.append({"label": execution["label"], "execution_id": execution_id,
            "algorithm_id": algorithm_id, "config": algorithm["config"],
            "config_type": algorithm["config_type"], "implementation": algorithm["implementation"],
            "provenance": execution["provenance"],
            "initial_state": execution["initial_state"], "status": execution["status"],
            "log_path": execution["log_path"], "step_count": len(steps),
            "warning_count": sum(row.get("level") == "warn" for row in logs),
            "local_solver_failures": sum(row.get("msg") == "Krylov linsolve did not converge." for row in logs),
            "step_durations": [row["data"]["duration"] for row in steps],
            "last_progress": steps[-1]["data"], "measurements": measurements,
            "final": final})
    order = ("als", "nnr_default", "nnr_slow_decay", "nnr_resumed")
    results.sort(key=lambda item: order.index(item["label"]))
    by_label = {item["label"]: item for item in results}
    assert set(by_label) == {"als", "nnr_default", "nnr_slow_decay", "nnr_resumed"}
    clean_solver_comparison = all(item["local_solver_failures"] == 0 for item in results)
    fixture = next(a for a in index["algorithms"].values() if a["label"] == "CDL fixture")
    if fixture["config"]["parameters"].get("normalize", False):
        assert clean_solver_comparison
    strong = by_label["nnr_slow_decay"]
    assert max(strong["final"]["effective_pair_ranks"]) == 1
    assert strong["final"]["relative_squared_error"] < 1e-12
    assert strong["final"]["rank1_relative_change"] < 1e-16
    # The control also fits the target, but has NOT eliminated the same channels.
    assert by_label["als"]["final"]["relative_squared_error"] < 1e-10
    assert by_label["als"]["final"]["max_second_ratio"] > 0.1
    assert by_label["als"]["final"]["rank1_relative_change"] > 1e-4
    assert by_label["nnr_resumed"]["algorithm_id"] == strong["algorithm_id"]
    assert by_label["nnr_resumed"]["initial_state"]["id"] in index["states"]
    for snapshot in index["snapshots"].values():
        assert snapshot["parent"]["id"] in index["states"]
        assert snapshot["format"] == "jld2-v1"
        assert snapshot["stored"] and snapshot["payload"].endswith(".jld2")
    paused = read_json(root / "paused_view.json")
    assert paused["session"]["status"] == "paused"
    assert paused["execution"]["status"] == "running"
    assert paused["latest"]["data"]["completed_updates"] == 3
    assert read_json(root / "completed_view.json")["status"] == "completed"
    assert not read_json(root / "tree.json")["truncated"]
    restart_check = read_json(root / "restart_check.json")
    assert restart_check["relative_squared_difference"] < 1e-18
    report = {"schema": 1, "store_id": index["id"], "json_only_checks_passed": True,
        "fixture": fixture["config"], "clean_solver_comparison": clean_solver_comparison,
        "snapshots": list(index["snapshots"].values()),
        "runs": results, "paused_view": paused, "restart_check": restart_check,
        "limits": [
            "One fixed random CDL fixture (seed 716); not a general convergence guarantee.",
            "Local singular spectra depend on gauge; rank-one projection change is also measured.",
            "Effective ranks use relative threshold 1e-6; allocated bond spaces are not resized.",
            "Step duration excludes callback measurement, snapshot I/O and log I/O.",
            "Execution records stay running while paused; live Session JSON needs an application export.",
            "Spectra are explicit measurement results, not automatically included in progress logs.",
            "A completed execution does not imply converged local solves: inspect warning counts.",
            "Lab configuration nodes expose stable type names and parameters; third-party values use their own field conversions.",
        ]}
    (root / "frontend_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# CDL NNR journey: JSON-only inspection", "",
        "All numbers below were read from JSON/JSONL, without loading tensors or Julia.", "",
        f"Clean solver comparison (no unconverged local solves): **{clean_solver_comparison}**.", "",
        "| Run | Site updates | Relative squared error | Max sigma2/sigma1 | Rank-one projection change | Solver failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in results:
        data = item["final"]
        lines.append(f'| {item["label"]} | {item["step_count"]} | {data["relative_squared_error"]:.3e} | '
                     f'{data["max_second_ratio"]:.3e} | {data["rank1_relative_change"]:.3e} | {item["local_solver_failures"]} |')
    lines += ["", "## What a frontend can see", "",
        "- Nested algorithm and solver parameters, configuration/run IDs, and input-state links.",
        "- Every site's sweep, cost, penalty, normalization scale, direction and duration.",
        "- Measurement spectra, effective ranks, physical errors and their parent state/run.",
        "- Separate measurement logs, snapshot metadata/paths and resumed-run provenance.",
        "- Paused and completed Session views when the application exports them.", "",
        "## Limits", ""] + ["- " + value for value in report["limits"]]
    (root / "frontend_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    inspect_store(Path(sys.argv[1]).resolve())
