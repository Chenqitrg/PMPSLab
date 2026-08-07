# WordReduction — Exploratory Prompt

> This is an exploratory cache-design prompt, not the PeriodicMPS canonical
> state specification. The authoritative finite-ring representation and
> projector gauge-fixing design live in `SPEC.md`.

I need a Julia submodule `WordReduction` with zero external dependencies. It solves the periodic product problem with updates: a sequence of elements with
associative `*`, where elements are mutated one at a time (`w[i] = a`), and products
over contiguous (possibly wrapping) ranges must stay queryable with minimal recomputation.

## Interface

Public interface:

Elements must support `*` (associative, not necessarily commutative). No separate `join` function.

```julia
w[i]        -> element       # access element at periodic index i
w[i] = a                     # assign element, invalidate affected cache
w[range]    -> product       # reduct: product over a periodic range

# Sweep: apply updaterule at each position in `order`, with caching
sweep(w::PeriodicWord, updaterule, order::Vector{Int})
    # For each i in order:
    #   W_i := w[i+1 : i-1]          # product of everything except w[i]
    #   w[i] = updaterule(W_i)     # assign back
    # Returns nothing; w is mutated in-place, log is accumulated internally.
```

Where `w` is a `PeriodicWord{T}` backed by `PeriodicVector{T}` (from MPSKit). `range` is
`i:j` (unit stride, possibly wrapping).

The `order` parameter is the key driver of the caching strategy. Examples:
- `1:N` — forward sweep, sliding window, O(1) amortized
- `vcat(1:N, N-1:-1:2)` — forward then backward (bidirectional), O(1) amortized
- Arbitrary sequence — falls back to segment tree, O(log N) per step

## Cost Model

Cost = **number of `*` calls**. Lower bound = 1 `*` per update (the newly generated
`w[i]` must be multiplied into at least one cached product). Storage is secondary but
tracked.

## Core Challenge

The module must choose the right cache representation **automatically** given `order`,
without the caller specifying a strategy. The tree structure is an *index* to find the
minimal-cache scheme — not the only representation.

Given `order`, the optimal cache should be computable before the sweep begins:
scan `order` to determine the access pattern, then select the optimal structure:

| `order` pattern | Structure | `*` calls per update |
|-----------------|-----------|-----------------|
| Sequential, no turn | Prefix/suffix chain | O(1) |
| Sequential, one turn | Two-stack deque | O(1) |
| Sequential, multiple turns | Segment tree | O(log N) |
| Arbitrary | Segment tree | O(log N) |

## Julia-Specific Constraints

- `PeriodicVector{T}` has native wrap-around (`w[0] == w[N]`).
- For small N (≤ 16), compile-time `Val{N}` can eliminate dispatch overhead.
- For `AbstractTensorMap`, `*` is matrix multiplication, costing O(D^3) and O(D^2) storage.
- For non-commutative `*`, ordered concatenation must be preserved.

## What Makes This Non-Trivial

1. After `w[i] = a`, only O(log N) nodes should be invalidated.
2. The tree index should pre-analyze `order` to compute the *optimal* cache scheme
   before the first query, not react lazily.
3. The identity element for `reduct` depends on the spaces being contracted
   (not `one(T)` for `TensorMap`).
4. When `order` switches direction mid-stream, useful cached segments from the previous
   direction should be retained, not discarded.

## Acceptance

- Works for `TensorMap` (via `*`) and any type with `*` defined.
- `w[range]` returns correct ordered product for wrapping ranges.
- Second query on an overlapping range hits cache, not full recompute.
- Single-element `w[i] = a` causes at most O(log N) segment invalidations.
- Zero dependencies beyond `PeriodicVector` from MPSKit (already available).