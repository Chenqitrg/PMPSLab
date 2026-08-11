# PeriodicMPS

Finite periodic matrix-product states and projector-based periodic gauge fixing.

## Project philosophy

PeriodicMPS is an experimental toolbox for tensor-network algorithm research.
Its goal is to provide a broad collection of low-level functions rather than
prescribe a single end-to-end workflow. Transfer construction, periodic
products and caches, gauge fixing, truncation, local solvers, sweeps, and
execution primitives are exposed as building blocks that researchers can
arrange and recombine when exploring new algorithms.

The project is intentionally aggressive about adding new algorithmic
capabilities. At the same time, it invests heavily in architecture: components
should have explicit invariants, narrow interfaces, and as little hidden policy
as possible, so that an ansatz, solver, sweep strategy, transfer representation,
or execution policy can be replaced independently and composed in new ways.
APIs may evolve as these primitives are refined; composability and clear
algorithmic foundations take priority over prematurely freezing a high-level
interface.

## Motivation

Tensor networks with closed loops are fundamental building blocks of more
complicated tensor-network methods, including projected entangled-pair states
(PEPS) and two-dimensional tensor-network renormalization (TNR). Understanding
the structure, gauge freedom, and numerical behavior of periodic MPS is
therefore an important step toward developing and reasoning about algorithms
for these more complicated networks.

## Periodic MPS and gauge fixing

A finite periodic MPS is not stored as an `InfiniteMPS` canonical unit cell.
The package keeps the original `PeriodicMPS` tensors and uses the same kind of
periodic QR/LQ fixed-point iteration as MPSKit's InfiniteMPS regularization to
compute left and right bond gauges. Projectors derived from those gauges are
then absorbed into the finite periodic state in place. `gaugefix!` first applies
MPSKit's exact local full-rank cleanup when the bond-space layout requires it;
the read-only `find_projectors` helper instead expects that cleanup to have
already been done.

```julia
mps = random_mps(ComplexF64, 6, 2, 4)
gaugefix!(mps, ProjectorCanonical())
```

There is deliberately no `CanonicalPeriodicMPS` storage type and no separate
finite-chain `canonicalize` operation. See [`SPEC.md`](SPEC.md) for the current
algorithm and invariants.

## Split status and cost reporting

The 2-to-1 `split` code keeps the mixed periodic transfer product oriented as
`⟨B|A⟩`. With an exact local solver, every update therefore agrees with the
complete finite-period cost and is non-increasing up to numerical roundoff.

`split` is the blocking convenience API:

```julia
alg = PeriodicSplit(
    SVDAnsatz(),
    CircularSweep(2 * length(A); maximal_sweep=4),
    CholeskySolve(),
)
B, log = split(A, alg)
```

Split environments use exact `DenseTransferMatrix` products by default. The
normal `⟨B|B⟩` and mixed `⟨B|A⟩` environments can instead select their transfer
representations independently:

```julia
truncated = TruncatedTransferPolicy(
    MPSKit.TruncationStrategy(maxrank=environment_rank),
)
alg = PeriodicSplit(
    SVDAnsatz(),
    CircularSweep(2 * length(A); maximal_sweep=4),
    KrylovSolve(GMRES()),
    SplitEnvironmentPolicy(truncated),
)
```

With a truncated normal environment, `KrylovSolve` retains the factored matrix
action. Direct solvers are also supported: they materialize the local factored
environment as a dense matrix immediately before solving. A truncated mixed
environment is contracted directly into the local right-hand side.

For controllable execution and per-site progress events, drive the same
algorithm through a run session:

```julia
session = start_run(
    initialize_run(A, alg);
    observer = event -> display(event.log.error),
)
wait(session)
B, log = run_result(session)
```

`pause!` and `cancel!` are non-blocking requests. To inspect or snapshot a
paused run manually, wait for its driving task before reading the numerical
state:

```julia
pause!(session)
wait(session)
run_status(session) == PMPSLab.LOCAL_PAUSED || error("run did not pause")
saved = snapshot(session.run)
write_checkpoint("split.jld2", saved)
```

The durable helpers perform the wait, safe-boundary check, snapshot, and atomic
file publication as one local operation:

```julia
pause_and_checkpoint!(session, "split.jld2")
resume!(session)  # reuses the in-memory run and its caches

cancel_and_checkpoint!(session, "cancelled-split.jld2")
```

After a process restart, supply the original target MPS and complete algorithm
configuration. Restore validates the target tensor spaces and rejects a
checkpoint whose ansatz, sweep, solver, or transfer policies do not match;
the caller remains responsible for supplying the correct target values:

```julia
saved = read_checkpoint("split.jld2")
session = start_run(restore_run(saved, A, alg))
```

The current cost is

```julia
last(log.site_logs).error
```

and, with dense environments, represents

```math
\lVert A-B\rVert^2 = \langle A|A\rangle + \langle B|B\rangle
                     - 2\operatorname{Re}\langle B|A\rangle.
```

With truncated working environments, `SiteLog.error` is the corresponding
approximate local cost. `squared_distance(A, B)` remains the exact dense
finite-period check.

`SiteLog.state.n` identifies the sweep. `log.sweep_time` is the accumulated
time spent inside completed site updates and excludes time spent paused.
