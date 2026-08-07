# PeriodicMPS

Finite periodic matrix-product states and projector-based loop gauge fixing.

A finite MPS on a ring is not stored as an `InfiniteMPS` canonical unit cell.
The package keeps the original `PeriodicMPS` tensors and uses the same kind of
periodic QR/LQ fixed-point iteration as MPSKit's InfiniteMPS regularization to
compute left and right bond gauges. Projectors derived from those gauges are
then absorbed into the finite ring in place.

```julia
mps = random_mps(ComplexF64, 6, 2, 4)
gaugefix!(mps, ProjectorCanonical())
```

There is deliberately no `CanonicalPeriodicMPS` storage type and no separate
finite-chain `canonicalize` operation. See [`SPEC.md`](SPEC.md) for the current
algorithm and invariants.
