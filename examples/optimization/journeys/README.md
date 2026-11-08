# CDL internal-loop journey

Start with [11_cdl_nnr.jl](../11_cdl_nnr.jl). This is a physical integration check,
not an extra optimizer or a general convergence guarantee.

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. examples/optimization/11_cdl_nnr.jl
python3 examples/optimization/journeys/read_cdl_json.py /the/path/printed/by/julia
```

Set `CDL_JOURNEY_OUTPUT` to a fresh directory to choose the artifact location.
The variables remain available when the example is included in the Julia REPL.

## Standard regression test

`Pkg.test()` includes [the CDL journey tests](../../../test/optimization/journeys/runtests.jl).
To focus the optimization tests on this journey (alongside the base MPS and transfer checks):

```julia
using Pkg
Pkg.test(; test_args=["journey"], julia_args=["-O0", "--threads=1"])
```

The regression uses 50 sweeps of ALS and slow-decay NNR, sharing this example's
physical diagnostics. It requires reconstruction error below `1e-10`, all NNR
pair ranks equal to one at relative tolerance `1e-6`, and rank-one projection
change below `1e-16`. ALS must retain its internal channels. The optimizer must
leave allocated bond dimensions and caller-owned inputs unchanged.

It also checks flushing inside each site callback, pause/resume, recovery from
sweep 25, and execution/measurement ownership. A Julia JSON reader checks a
separate copy containing only JSON/JSONL files, including per-step cost, duration,
site, sweep and penalty. No Python or permanent output directory is required;
temporary files are removed automatically. The 100-sweep example remains useful
for inspecting saved artifacts and comparing the default NNR schedule as well.

## Physical question

A four-site CDL target with corner dimension 2 contains an internal closed
correlation loop. Its initial SVD split already reconstructs the physical state
to numerical precision. Low reconstruction error therefore does not demonstrate
that the redundant loop has disappeared.

The example compares ordinary ALS, default NNR, and NNR with
`initial=0.01, factor=0.95, minimum=1e-7`, all for 100 circular sweeps. All runs use
the same initial tensors and GMRES settings. The generated target is uniformly
normalized for **all** branches, so their absolute solver tolerances have the same
scale. This scalar normalization cannot remove a correlation loop. There is no
gauge filtering and no rank truncation inside the optimization.

[cdl_diagnostics.jl](cdl_diagnostics.jl) measures:

- Each site's singular spectrum across the inserted pair bond, normalized to
  its largest singular value; an effective rank uses relative threshold `1e-6`.
- The actual relative squared error from the small dense physical state. This
  avoids cancellation in the quadratic cost formula at very small errors.
- The physical change if every inserted pair bond keeps only its leading local
  singular channel. This diagnostic projects a **copy** after the update and
  never feeds it into the algorithm.
- Allocated bond dimensions, which the optimizer does not resize.

Local spectra depend on gauge. The projection diagnostic and reconstruction
error must be read together with them. These are local singular spectra, not
CFT scaling dimensions. Even rank-one spectra mean *effective* removal within
numerical tolerance, not automatic storage-space compression.

## Storage and frontend journey

The example generates and snapshots the input under a fixture configuration,
then links each optimization execution to that input state. It stores one
measurement at the initial state and after sweeps 1, 5, 10, 20, 50 and 100.
Every site update is separately written to the execution's JSONL log. The callback
reads that file during execution and verifies that its latest step is already
flushed and has the correct execution owner.

The slow-decay run pauses after site 3. The application exports its live Session
view, then resumes it. Snapshots at sweeps 50 and 100 include the NNR auxiliary
and dual tensors. A new execution restores sweep 50, finishes the remaining 400
site updates, and is compared with the uninterrupted result.

[read_cdl_json.py](read_cdl_json.py) uses only Python's standard library and
JSON/JSONL files: it does not import Julia or open a tensor/JLD2 payload. It follows
configuration → execution → state → measurement links, verifies step sequences,
reads spectra, checks ownership of both execution and measurement logs, and
checks the application-exported pause/resume views. It writes:

- `frontend_summary.json`: machine-readable collected evidence.
- `frontend_report.md`: a compact human-readable comparison and limitations.

Tensor-level equality on restart is checked by Julia; the JSON reader can inspect
that exported check and verify links, but cannot independently recompute tensors.

## Boundaries this test makes visible

- Progress and measurement data are separate. A frontend can plot spectra only
  after the application schedules and stores those measurements.
- `ExecutionRef.status` remains `running` during a pause. To show `paused`, an
  application must publish `ExperimentLab.describe(session)`; reading the index
  alone is insufficient. The example exports `paused_view.json` and
  `completed_view.json` to make this explicit.
- Each Lab configuration node carries a stable `type` and a `parameters` object,
  including empty strategies. Third-party values use their own field conversions.
  This identifies configurations but is not a frontend form schema.
- A completed execution can contain unconverged local solves. The reader reports
  warnings and solver-failure counts instead of treating `completed` as a
  numerical-success certificate.
- `duration` is the numerical site-update duration. It excludes callback
  measurements, snapshot writes, and log I/O; summing it is not wall-clock runtime.
- Snapshot references and binary paths are visible in JSON. Tensor contents
  remain Julia-only, intentionally.
