# PMPSLab review examples

Each file covers one topic and runs independently. No earlier example needs to
run first. The `@test` assertions check the behavior just demonstrated.

| Order | File | Question |
| --- | --- | --- |
| 1 | [01_site_transfer.jl](01_site_transfer.jl) | How does `transfer_at` construct a transfer from an MPS site? |
| 2 | [02_periodic_product.jl](02_periodic_product.jl) | What does `PeriodicTransferMatrix` store, and how is a full-period product formed? |
| 3 | [03_overlap.jl](03_overlap.jl) | How does `expectation(bra, ket)` compute an overlap? |
| 4 | [04_cache.jl](04_cache.jl) | How does an MPS connect to the periodic cache you have already reviewed? |
| 5 | [05_cache_update.jl](05_cache_update.jl) | How do you invalidate and repair a cache after changing the MPS? |
| 6 | [06_transfer_policy.jl](06_transfer_policy.jl) | How does a policy differ from a numerical transfer value? |
| 7 | [07_norm_and_normalization.jl](07_norm_and_normalization.jl) | How do norms, squared distances, and the two normalization interfaces work? |
| 8 (advanced) | [08_paired_mps.jl](08_paired_mps.jl) | How does a two-physical-leg target pair with a single-physical-leg MPS? |

Start with 01–03, then read 04–07. Example 08 covers the mixed representation used
by the split algorithm and can be read last.

Run from this repository:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/environments/01_site_transfer.jl
```

In an editor, select `PMPSLab` as the Julia project. If your REPL
loaded the old packages before the extraction, restart Julia before running these
examples. Each file ends with `nothing` to suppress the final return-value display.

`rand(PeriodicMPS, ...)` uses the default random number generator; an explicit
`rng` is optional. Numerical values in the random examples vary between runs,
but the checked relationships remain the same.

## Interface ownership

- `PMPSLab`: MPS data, construction, and full-state contraction.
- `TransferMatricesLab`: individual dense/factored transfers and multiplication.
- `PeriodicProductsLab`: generic cache interfaces such as `build_cache`, `environment`, `invalidate!`, and `rebuild!`.
- `PMPSLab`: PMPS-specific `build_transfer_cache`, `rebuild_transfer_cache!`, and directional rebuild adapters, plus `transfer_at`, representation policies, overlaps, and norms.

`environment(cache, site)` is the ordered transfer product excluding the current
site, starting at the next site. `expectation(bra, ket)` means `<bra|ket>` here;
no operator is inserted.

Generic `build_cache`/`rebuild!` operate on transfer factors or getters. The named
transfer-cache adapters accept `PeriodicMPS` inputs. Restart an existing Julia
session after this API rename so previously loaded methods do not linger.
