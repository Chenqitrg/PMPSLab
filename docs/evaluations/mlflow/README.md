# Direct MLflow evaluation

This is a disposable evaluation of existing APIs, not a new framework or adapter.
It does not depend on AlgorithmConfigLab, AlgorithmDataLab, AlgorithmLogLab, or
ExecutionLab. All examples are top-level Julia code for REPL inspection.

Current direction: adoption is deferred because the user prefers native Julia
without a Python server. See [the DrWatson evaluation](../drwatson/README.md).
The feasibility results below remain valid; they are not a migration decision.

## Result: suitable for adopting a simpler workflow

Evaluated on 2026-11-06 with Julia 1.12.5, MLFlowClient 0.10.0, MLflow 3.14.0,
JLD2 0.6.7, TensorKit 0.17.3, LoggingExtras 1.2.0, and LoggingFormats 1.6.0.
The server was restarted after writing, and verification ran in a fresh Julia
process. All 30 checks passed:

| Check group | Passed |
| --- | ---: |
| Configuration, per-step history, typed TensorKit snapshots, child measurements, JSONL logs, checkpoint reference, and copied memory-only measurement | 23 |
| Failure before any state is produced, with retained diagnostics | 4 |
| Small artifact upload, download, and run-scoped listing over HTTP | 3 |

The tensor round trip used a small `TensorMap{ComplexF64,ComplexSpace,1,1}`.
This is not a benchmark or evidence for every symmetry, GPU array, or distributed
storage backend. No MPS optimization or LoopTNR calculation was run.

Recommendation: adopt MLflow runs/metrics/artifacts as the tracking model, JLD2
for Julia objects, and existing Logging packages for JSONL. Do not recreate our
Ref tree or add a compatibility adapter. AlgorithmDataLab and AlgorithmLogLab can
be retired once their consumers use this workflow. AlgorithmConfigLab is not
replaced by MLflow itself: ordinary Julia types, Accessors, and JSON/JLD2 cover
configuration construction, updates, and persistence if graphical tree input is
no longer required. ExecutionLab's interactive controls remain a separate choice.

Keep application code for choosing what to save, calculating measurements,
copying mutable inputs, waiting for background tasks, and marking failures.
MLflow records a run; it does not automatically execute or supervise arbitrary
Julia code. Input tags are conventions rather than enforced provenance edges.

No existing Lab implementation has been removed during this evaluation.

## Start the local server

From this directory, create a separate Python environment (Python 3.12 was used):

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'mlflow==3.14.0'
mkdir -p workspace
MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false OPENBLAS_NUM_THREADS=1 \
  .venv/bin/mlflow server --host 127.0.0.1 --port 5097 --workers 1 \
  --backend-store-uri sqlite:///workspace/mlflow.db \
  --artifacts-destination workspace/proxy-artifacts
```

Leave that terminal open. The UI is at <http://127.0.0.1:5097>.
Background GenAI jobs are disabled because this evaluation only needs tracking.
The server is local; no cloud account or subscription is involved.

## Review in order

| File | What to inspect |
| --- | --- |
| `00_minimal.jl` | Start here: one run, one parameter, three cost values, and completion |
| `01_run.jl` | Configuration, per-step metrics, ordinary `@info`, two TensorKit snapshots, and a checkpoint reference |
| `02_measurements.jl` | A measurement child run, typed and JSON results, and an asynchronous memory-only measurement |
| `03_reopen.jl` | Reopen the run, query its history and child runs, and restore Julia objects |
| `verify.jl` | Separate executable checks, a client timestamp probe, and failure before any snapshot exists |

Each example activates this standalone environment. For selected-line execution,
run its setup lines first. The numbered examples can run in separate Julia sessions.
Use one Julia/precompilation thread for this tiny evaluation:

```sh
JULIA_NUM_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no 01_run.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no 02_measurements.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no verify.jl
```

The Julia client's URL defaults to the server above. `MLFLOW_REVIEW_URL` can
override it, and `MLFLOW_REVIEW_WORKSPACE` can select another absolute workspace.
The last parent run ID is stored in `workspace/last-run.json`.
The checked dataset is retained there locally and ignored by Git. The evaluation
server is stopped after verification; start it with the command above to inspect
the retained runs. Running 01 again creates a new experiment/run.

## Accepted model

- An MLflow run replaces our execution record.
- Parameters searchable in the UI are strings. The complete nested configuration
  is saved as JSON and JLD2. Ordinary Julia constructors and Accessors can build
  or update configurations; MLflow does not provide the graphical tree macro.
- A step is a metric step and a snapshot filename, not a StateRef node.
  Save each step as shown, or write only the selected/final step; no retention
  framework is required for that choice.
- A measurement is a child run, with ordinary tags describing its input.
- `runs:/<run-id>/<artifact-path>` identifies an existing snapshot. A checkpoint
  tag and measurements can refer to it without saving another copy.
- A small result has a typed JLD2 representation for Julia and JSON for inspection.
- JSONL logs use Julia Logging, LoggingExtras, and LoggingFormats directly,
  in the appropriate run directory. Fields remain machine-readable numbers.
  There is no custom event registry or logger implementation.

## Storage and concurrency limits

This example deliberately uses a local/shared filesystem artifact location.
The server and Julia must see the same absolute paths. JLD2 writes directly into
the artifact directory returned by MLflow; no upload staging file or complete
in-memory file byte array is needed. This is not a portable recipe for S3 or a
server with an unrelated filesystem. File URI handling here is intentionally
limited to the local paths used by the example.

The asynchronous example uses `@async` and a copied tensor. It verifies ownership
and waiting, not CPU-parallel measurement throughput. `fetch` propagates task
errors. Scheduling and tensor copying remain Julia/application responsibilities.
Only one task writes each output file. Readers should consume snapshots after
the write completes; this example does not implement atomic snapshot publication.

Reference tags are conventions, not foreign keys. MLflow does not enforce our
former pin, retention, closed-provenance, or snapshot-deduplication invariants.
Changing `checkpoint` changes a tag; it does not copy or delete the snapshot.
The Julia definitions and a compatible package environment are still needed to
restore scientific objects.

## Released Julia client caveats

Evaluated client: MLFlowClient 0.10.0. Its `logmetric` default uses seconds from
local `now()`, although MLflow expects Unix milliseconds. The walkthrough passes
`timestamp=round(Int, 1000time())` explicitly. `verify.jl` separately probes the
unmodified default. This is not patched or hidden behind an adapter.

Its ordinary artifact upload/download API accepts/returns a whole `Vector{UInt8}`
and operates on artifact-proxy paths. It is not a high-level streaming file API
automatically rooted at a run. Prefer the shared-file approach evaluated here for
large local tensors, or evaluate MLflow's official Python SDK separately before
adopting remote object storage. Julia `@info` is not automatically sent to MLflow.

## Sources

- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow artifact storage](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/)
- [MLFlowClient source](https://github.com/JuliaAI/MLFlowClient.jl)
- [Client metric timestamp implementation](https://github.com/JuliaAI/MLFlowClient.jl/blob/main/src/services/logger.jl)
- [Client artifact API](https://github.com/JuliaAI/MLFlowClient.jl/blob/main/src/services/artifact.jl)
