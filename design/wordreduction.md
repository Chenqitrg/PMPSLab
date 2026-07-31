# WordReduction — Prompt

I need a Julia submodule `WordReduction` with zero external dependencies. It solves the periodic product problem with updates: a sequence of elements with
associative `join`, where elements are mutated one at a time (`w[i] = a`), and products
over contiguous (possibly wrapping) ranges must stay queryable with minimal recomputation.

## Interface

Three things are public:

```julia
join(a, b) = a * b          # associative binary operation, overloadable per type
w[i]        -> element       # access element at periodic index i
w[i] = a                     # assign element, invalidate affected cache
w[range]    -> product       # reduct: product over a periodic range
```

Where `w` is a `PeriodicWord{T}` backed by `PeriodicVector{T}` (from MPSKit). `range` can be
`i:j` (unit stride, possibly wrapping) or `(start, length)`.

## Cost Model

The only cost that matters is **number of `join` calls**. Storage is secondary but should
be tracked. The goal: given a stream of `w[range]` queries interleaved with `w[i] = a`
assignments at a *known, predictable rate* (typically: modify one element, then query a
set of ranges that depend on it, repeat in a sweep), minimize total joins.

## Core Challenge

Different access patterns have different optimal strategies:

- A single L→R sweep queries expanding prefixes: `[1:1], [1:2], ..., [1:N]`. Optimal is a
  prefix chain (1 join per step, O(N) storage).
- Bidirectional sweeps alternate L→R and R→L. Prefix/suffix chains conflict.
- Random access, all-pairs, or unknown patterns need something more general.

The module should choose the right cache representation **automatically**, without the
caller specifying a strategy. The tree structure should serve as an index to *find* the
minimal-cache scheme for the observed query pattern, rather than being the only
representation.

## Julia-Specific Constraints

- `PeriodicVector{T}` has periodic indexing (`w[0] == w[N]`, `w[N+1] == w[1]`), so
  wrap-around ranges are native.
- For small N (≤ 16, common in LoopTNR), compile-time knowledge can eliminate dispatch
  overhead.
- The structure should support any `T` where `join(a::T, b::T)::T` is defined.
- For `T <: AbstractTensorMap`, `join` defaults to `*`, which costs O(D^3) and produces
  O(D^2) storage.

## Update Rule

In a sweep, the update `w[i] = a` is followed by queries that depend on `i` (e.g., `w[1:i]`
or `w[i:N]`). The update pattern is deterministic and known: elements are visited in order
(L→R or R→L), modified one at a time, and queries immediately reflect the new value.
This predictability is the key to the caching strategy.

## What Makes This Non-Trivial

1. The optimal cache for a L→R sweep (prefix chain) is wrong for R→L. Switching direction
   mid-stream should not throw away useful work.
2. After `w[i] = a`, only O(log N) nodes should be invalidated, not the whole cache.
3. The tree that stores cached products should double as the decision structure for
   "which segments to keep vs recompute," given the known update trajectory.
4. The identity element needed by `reduct` for non-commutative types is not `one(T)` — it
   depends on the spaces being contracted. This must be handled generically.

## Acceptance

- Works for `TensorMap` (via `*`) and any type with `join` defined.
- `w[range]` returns correct ordered product for wrapping ranges.
- Second query on an overlapping range hits cache, not full recompute.
- Single-element `w[i] = a` causes at most O(log N) join invalidation.
- Zero dependencies beyond `PeriodicVector` from MPSKit (already available).
