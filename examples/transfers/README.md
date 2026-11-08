# Review examples

Each file runs independently or section by section in a Julia editor. Imports
are limited to the symbols needed for the feature being demonstrated.

1. [01_construction.jl](01_construction.jl): construct a transfer from one site tensor and inspect its domain and codomain.
2. [02_bra_ket.jl](02_bra_ket.jl): bra/ket ordering and conjugation.
3. [03_composition.jl](03_composition.jl): transfer multiplication and application to a bond tensor.
4. [04_truncation.jl](04_truncation.jl): SVD factors, retained dimension, and truncation error.

Run from this repository:

```sh
julia --project=. examples/transfers/01_construction.jl
```

`TensorMap(E)` returns the stored data for a dense transfer and computes
`u * s * vᴴ` for a truncated transfer. It does not convert to an ordinary Array
These examples cover individual transfers; periodic environments and optimization
are reviewed separately in the same package.
