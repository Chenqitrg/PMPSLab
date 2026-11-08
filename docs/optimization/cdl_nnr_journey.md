# CDL NNR journey: observed results

Run date: 2026-11-07. Numerical implementation: PMPSLab commit `e916f18`.
Julia 1.12.5, one Julia thread, one BLAS thread. No package versions changed.
The reproducible entry point is [example 11](../../examples/optimization/11_cdl_nnr.jl); see its
[journey guide](../../examples/optimization/journeys/README.md) for the storage/reader protocol.

A smaller version now runs automatically in `Pkg.test()` via
[test/journeys/runtests.jl](../../test/optimization/journeys/runtests.jl): 50 sweeps of ALS and
slow-decay NNR, restart from sweep 25, and JSON-only frontend assertions. Its
reconstruction threshold is `1e-10`; the measurements below describe the longer
100-sweep example, not the regression's stopping point.

## What was actually tested

A random four-site CDL MPS, corner dimension 2, seed 716, was uniformly
normalized and split into eight sites with untruncated SVD initialization.
The initial state already reproduces the target, but its largest local
second/first singular-value ratio is 0.5466: it contains redundant internal
channels. No gauge filtering or bond compression is used by any optimization run.

All branches share the initial tensors, 100 circular sweeps and GMRES
(`tol=1e-12`, `maxiter=100`). The slow-decay NNR uses
`initial=0.01`, `factor=0.95`, `minimum=1e-7`, preserving `mu=xi^2`.
The default NNR uses the package defaults. Measurement values below come from
stored JSON, independently collected by the Python reader.

| Run | Relative squared reconstruction error | Max local sigma2/sigma1 | Effective pair ranks | Relative squared change after rank-one projection |
| --- | ---: | ---: | --- | ---: |
| ALS | 2.616e-31 | 5.466e-1 | all 2 | 3.074e-2 |
| Default NNR | 9.420e-24 | 7.403e-3 | 2, 2, 1, 1, 1, 1, 1, 1 | 2.938e-27 |
| Slow-decay NNR | 8.275e-17 | 2.069e-11 | all 1 | 3.514e-31 |

Effective rank uses a relative singular-value threshold of `1e-6`. The rank-one
projection is a measurement on a detached copy, never an optimization update.
Reconstruction error is computed by dense contraction and subtraction for this
small fixture, not by subtracting nearly equal quadratic cost terms.

The stronger NNR removes all tested internal channels to numerical tolerance
while preserving the physical state. Default NNR also produces a state whose
rank-one projection is effectively unchanged, but leaves small local tails at
the first pair. Ordinary ALS stays at the accurate but redundant initial state.
This demonstrates why reconstruction error alone is not a loop-removal test.
Local singular spectra are gauge-dependent, so the physical projection check is
reported alongside them. This is one fixture, not a general convergence theorem
or a parameter recommendation for every tensor.

Allocated bond dimensions remain `[8, 4, 8, 4, 8, 4, 8, 4]`: effective channel
removal does not resize the tensor spaces. No automatic memory compression is
claimed.

## Execution, storage and JSON-only checks

The journey produced four configurations, five execution records (including
input generation and restart), 23 states, 22 measurements and three snapshots.
It recorded 2,800 site updates across the three runs and the restarted run.
Each original-run callback read the JSONL log during execution and verified its
latest step and execution owner. Measurement logs retained their own owners.

The slow-decay run paused after site 3. Its exported live Session JSON said
`paused`, while its persistent execution record correctly remained `running`.
The run continued and wrote snapshots at sweeps 50 and 100. A new execution
loaded the sweep-50 snapshot and performed updates 401–800. Its final dense
state matched the uninterrupted run exactly in this experiment (reported
relative squared difference 0.0).

The Python consumer followed all ownership links and read configuration values,
site/sweep/direction, local cost, penalty, normalization scale, step duration,
singular spectra, effective ranks, measurement results, logs and snapshot paths.
It succeeded again on a directory containing **only JSON/JSONL files**, with
zero JLD2 payloads. The consumer verifies published numerical results and links;
it cannot independently recompute tensor quantities without those tensors.

## Gaps exposed by the journey

1. **Completion is not a numerical convergence certificate.** An exploratory run
   on the original, unnormalized fixture produced 800 unconverged ALS local
   solves and 1,876 warning rows while the execution still completed. The target
   squared norm was approximately 19,989, so an identical absolute Krylov
   tolerance did not make a clean normalized/unregularized comparison. That
   exploratory record was retained. The normalized comparison above has zero
   solver failures and zero warnings in every branch. The frontend reader
   reports both counts. Residuals and solver convergence are currently warning
   messages, not dedicated structured progress fields.
2. **Persistent execution status does not expose pause state.** A live frontend
   must obtain `ExperimentLab.describe(session)` from the application. The
   example publishes `paused_view.json` and `completed_view.json` explicitly.
3. **Configuration identity is explicit, but JSON is not a full frontend schema.**
   Each Lab configuration now has a stable `type` and a `parameters` object,
   including stateless strategies. Third-party fields retain their own conversion
   rules. Allowed ranges and selectable alternatives are not described here.
4. **Scientific diagnostics must be scheduled explicitly.** Spectra are stored
   as measurements associated with states; progress logs do not invent them.
   These measurements are local singular spectra, not CFT spectra.
5. **Step duration has a precise, limited meaning.** It covers the numerical
   update, not callback measurements, snapshot writes, log I/O or time paused.

This work adds a journey example, diagnostic measurements and a JSON consumer.
It does not change the optimizer or ExperimentLab's core APIs.
