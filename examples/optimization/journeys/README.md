# CDL internal-loop journey

Start with [11_cdl_nnr.jl](../11_cdl_nnr.jl). This is a physical integration check,
not an extra optimizer or a general convergence guarantee.

Use an experiment environment containing PMPSLab, DrWatson, JLD2 and JSON.

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=/path/to/experiment examples/optimization/11_cdl_nnr.jl
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

The test also saves a numerical checkpoint at sweep 25, restores it and compares
the final tensors with the uninterrupted result. A JSON-only reader checks
site/sweep order, costs, penalty and physical measurements. Temporary files are
removed automatically. The 100-sweep example also compares the default NNR schedule.

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

## Storage and inspection

The application runs an A.I. loop and collects progress and scientific
measurements explicitly. DrWatson's `tagsave` stores results with source metadata;
JLD2 stores a numerical checkpoint at sweep 50. Restoring it finishes the remaining
400 updates and must reproduce the uninterrupted tensors.

The application writes a compact JSON summary for each run.
[read_cdl_json.py](read_cdl_json.py) reads those summaries with Python's standard
library, without loading Julia or tensor payloads. Tensor equality on restart is
checked by Julia; JSON consumers can only inspect the exported measurements.

There is no session database, automatic measurement scheduling or persistent pause
status. The application decides where to stop, what to save and how to name runs.
Step duration covers numerical work and excludes measurements and storage.
