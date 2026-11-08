# Review periodic MPS optimization

From the PMPSLab directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/optimization/01_simple_split.jl
```

Each example runs independently, uses English comments, and ends with `nothing`
so including it does not print a testset object.
Variables remain available in the REPL. The default PMPSLab API uses short names; specialized policies are explicitly imported; other Lab packages are qualified so each call shows its owner.

## Start with the numerical workflow

Read **01 → 03 → 07 → 08**, then the two execution examples.

| Component | Its contribution to a split | Where to see it |
| --- | --- | --- |
| PMPSLab representation | Target and approximation tensors | 01: `rand(PeriodicMPS, ...)` |
| LatticeIteratorLab | Visit order supplied as `sweep` | 01 and 04: `CircularSweep` / `ZigzagSweep` |
| TensorLinearSolveLab | Local equation solver supplied as `solver` | 01 and 03: `CholeskySolve` / `LUSolve` |
| PMPSLab transfers | Dense or factored numerical transfer values | 03 and 07: inspect the actual transfer type |
| PMPSLab environments | Construct transfers/caches from MPS; evaluate distance | 03: `transfer_at`; 07: `TruncatedTransferPolicy` |
| PeriodicProductsLab | Store partial products and provide a site's environment | 03: inspect the actual split cache and call `environment` |
| PMPSLab gauge operations | Gauge fixing applied during initialization | 08: explicit `gaugefix!` and equivalent `GaugeInitialization` |
| ExperimentLab | Drive steps, pause/resume, own logs and persist snapshots | execution/01 and execution/02 |

PMPSLab supplies the local split equations and decides when to update each
cache. You pass policies to it; you do not need to drive its caches manually.
Example 03 deliberately inspects the internal workspace to make that boundary
visible. The ordinary application entry point remains `split(target, algorithm)`.

## Individual features

| Example | One question to review |
| --- | --- |
| [01 Simple split](01_simple_split.jl) | How do MPS data, a sweep and a solver form one algorithm? |
| [02 Initialization](02_initial_approximation.jl) | Which bonds does SVD truncation change? |
| [03 One site update](03_one_site_update.jl) | How do transfer values, caches and the local solve cooperate? |
| [04 Sweep order](04_sweep_order.jl) | Does the split algorithm follow the order supplied by LatticeIteratorLab? |
| [05 Stopping and cost](05_stopping_and_cost.jl) | When is convergence checked? |
| [06 Existing approximation](06_existing_approximation.jl) | What is copied, and what does `copy_initial=false` share? |
| [07 Exact error](07_exact_error.jl) | How does a transfer policy affect the local estimate? |
| [08 Gauge initialization](08_gauge_initialization.jl) | How does split initialization reuse gauge fixing? |
| [09 Nuclear norm](09_nuclear_norm.jl) | How do nested NNR settings and automatic normalization work? |
| [10 NNR restart](10_nuclear_norm_restart.jl) | How are auxiliary tensors preserved by ordinary ExperimentLab snapshots? |
| [11 CDL journey](11_cdl_nnr.jl) | Does NNR remove redundant internal channels, and can a JSON-only reader verify the run? |
| [12 Configuration JSON](12_configuration_json.jl) | How are all nested Lab configurations named and restored? |

The numerical examples run synchronously; controlled execution uses ExperimentLab. For the source reading
order, see [the focused review guide](../../docs/optimization/split_review.md).

Example 11 records three 100-sweep comparisons, pauses at site 3, and resumes a
saved run from sweep 50. Its [journey guide](journeys/README.md) explains the
physical checks and the independent Python JSON reader. It runs separately
from the lightweight examples and is not a benchmark of runtime performance.

## Optional execution

[Controlled execution](execution/01_controlled_execution.jl) leaves the session
paused after site 2, before site 3 starts. Inspect `state` and `session` in the
REPL, then execute the commented `resume!` and `fetch` lines to finish.
It uses the PMPSLab root environment, like the numerical examples:

```sh
julia --project=. -i examples/optimization/execution/01_controlled_execution.jl
```

`execution/02_recorded_split.jl` demonstrates ExperimentLab-owned JSONL history,
a single-copy numerical capture, and restart from disk.
