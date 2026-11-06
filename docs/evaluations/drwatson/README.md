# DrWatson: an entirely local Julia workflow

This evaluates DrWatson 2.19.1 directly, without a Python environment, server,
database service, Lab packages, or compatibility adapters. Dependencies are
installed through Julia's package manager; computation and storage then happen
in the current Julia process and local files.

## Verified result

On 2026-11-06, the minimal example and full workflow ran successfully with Julia
1.12.5, DrWatson 2.19.1, JLD2 0.6.7, TensorKit 0.17.3, and DataFrames 1.8.2.
Verification ran in a fresh Julia process: 16 scientific-workflow checks and
2 cache-semantics checks passed. Nested NamedTuples and a small dense complex
TensorMap were restored, JSON logs retained step/cost fields, and the measurement
and checkpoint used the same snapshot path. Result collection read only small
summary files. No heavy numerical algorithms or server processes were run.

This supports adopting existing Julia packages for these functions, provided we
accept a small application-level directory/metadata convention. It does not show
that DrWatson alone replaces the full run/provenance system.

## Start here

Open `00_minimal.jl` in VS Code and run the file. For selected-line execution,
run the setup and imports first. No service needs starting.

```julia
settings = (chi=16, tolerance=1e-10)
result = (cost=0.001,)
tagsave(file, Dict("settings" => settings, "result" => result))
saved = wload(file)
```

The complete example supplies a new directory, a parameter-derived filename,
and the Git repository path. `result` is where the actual algorithm's output goes.
The outer loaded container is a dictionary; the stored NamedTuple settings and
result retain their Julia types.

| File | Purpose |
| --- | --- |
| `00_minimal.jl` | Save settings and a result, attach Git information, and load them |
| `01_run_and_measure.jl` | Two tensor snapshots, per-step JSONL logs, one measurement, and a small summary |
| `02_reopen_and_collect.jl` | Restore objects in a fresh process and collect summaries into a DataFrame |
| `verify.jl` | Check the scientific workflow and optional cache semantics |

Run 01 before 02 or verification. Variables remain available in the REPL.
Data are kept under `workspace/`, ignored by Git. Use a single compilation thread
for this small example:

```sh
JULIA_NUM_THREADS=1 JULIA_IMAGE_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no 01_run_and_measure.jl
JULIA_NUM_THREADS=1 JULIA_IMAGE_THREADS=1 JULIA_NUM_PRECOMPILE_TASKS=1 OPENBLAS_NUM_THREADS=1 \
  julia --startup-file=no verify.jl
```

## What is provided, and what is our convention?

| Responsibility | Provider |
| --- | --- |
| Parameter-derived filename | DrWatson `savename` |
| Save/load dictionaries and Julia objects | DrWatson `wsave`/`wload`, using JLD2 |
| Record Git commit and optionally a patch | DrWatson `tagsave` |
| Parameter combinations | DrWatson `dict_list` |
| Collect result files into a table | DrWatson `collect_results` with DataFrames |
| JSON logs and logger scopes | Logging, LoggingExtras, LoggingFormats |
| Run IDs, directory layout, parent/input references | Explicit application convention in the example |
| Run status and exception handling | Ordinary application code |
| Measurement computation, copying, asynchronous execution | Ordinary Julia/application code |

Our example convention is:

```text
workspace/
  runs/<run-id>/
    run.jld2                  # Nested settings, Git information, status, checkpoint path
    config.json
    run.jsonl                 # Includes the algorithm-start message and per-step cost
    snapshots/step-1.jld2
    snapshots/step-2.jld2
    measurements/<measurement-id>/
      result.jld2             # Measurement settings, result, parent ID, input path
      result.json
      measurement.jsonl
  summaries/<run-id>.jld2      # Small data only, for collect_results
```

Paths stored in the records are relative to the workspace or run directory.
The measurement and checkpoint refer to one snapshot file, without another copy
of the tensor. These paths are not typed references or enforced foreign keys.
DrWatson does not automatically intercept the algorithm or associate its outputs.

## Choices that matter for our use

- **Independent repeats:** create a fresh run directory. Do not use the parameter
  filename as the identity of a run: identical settings may be executed again.
- **Optional caching:** `produce_or_load` reuses an existing file by name. It does
  not automatically compare the current algorithm implementation with saved code.
  The verification deliberately changes the callback and confirms the old result
  is reused. Use this only when reuse is intended and the cache key is adequate.
- **Large tensors:** keep small summaries separate. `collect_results` can load
  result files before selecting columns; a whitelist is not a substitute for
  separating tensor payloads from the files being collected.
- **Logging:** JSONL is supplied by existing Logging packages, not DrWatson.
  There is no built-in metric-history service or web interface.
- **Provenance:** `tagsave` records Git information, not a complete environment
  snapshot. This example suppresses the dirty-repository warning but retains the
  dirty marker. Saving a tracked patch is optional; it does not capture untracked
  files. Julia type definitions and compatible dependencies remain necessary for
  object restoration.
- **Persistence:** this evaluation uses synchronous writes and one writer per file.
  It does not establish crash-safe publication, concurrent writers, pin/retention
  semantics, or asynchronous disk throughput.

DrWatson can replace generic saving, naming, Git tagging, parameter sweeps, and
result aggregation code. It is not a complete run/measurement lifecycle system.
Accepting a small directory/metadata convention avoids rebuilding our Ref layer,
but that convention remains our responsibility. No existing packages are removed
or merged by this evaluation.

## Sources

- [Project scope](https://juliadynamics.github.io/DrWatson.jl/stable/)
- [Saving and optional caching](https://juliadynamics.github.io/DrWatson.jl/stable/save/)
- [Parameter sweeps and result collection](https://juliadynamics.github.io/DrWatson.jl/stable/run&list/)
