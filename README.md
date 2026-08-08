# PeriodicMPS

Finite periodic matrix-product states and projector-based periodic gauge fixing.

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

`split` accepts an optional synchronous callback:

```julia
alg = PeriodicSplit(
    SVDAnsatz(),
    CircularSweep(2 * length(A); maximal_sweep=4),
    CholeskySolve(),
)
B, log = split(A, alg; callback = log -> display(last(log.site_logs).error))
```

After every complete sweep it calls `callback(log)`. The current cost is simply

```julia
last(log.site_logs).error
```

and represents

```math
\lVert A-B\rVert^2 = \langle A|A\rangle + \langle B|B\rangle
                     - 2\operatorname{Re}\langle B|A\rangle.
```

The callback adds no contraction or copying: it receives the same `SplitLog`
that is returned by `split`. `SiteLog.state.n` identifies the sweep and
`log.sweep_time` contains the elapsed run time.
