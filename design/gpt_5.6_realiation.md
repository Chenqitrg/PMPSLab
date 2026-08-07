> **Status: non-authoritative generated draft.** This file is retained only as
> exploratory material for the independent `WordReduction` idea. It is not the
> PeriodicMPS package specification and none of its code should be copied into
> the finite-ring projector implementation without separate review. See
> `SPEC.md` for the authoritative design.

Below is a self-contained implementation. It uses only Base plus `MPSKit.PeriodicVector`. Current MPSKit documentation defines `PeriodicVector{T}` as the one-dimensional alias of `PeriodicArray{T,1}` and confirms that it supports periodic integer indexing. ([quantumkithub.github.io](https://quantumkithub.github.io/MPSKit.jl/stable/lib/lib/?utm_source=openai))

A few semantics need to be explicit:

- `w[i:j]` means the nonempty forward periodic arc from `i` to `j`, spanning at most one period.
- Thus, for \(N=5\), `w[4:2]` computes `w[4] * w[5] * w[1] * w[2]`.
- Since an associative `*` alone does not define an empty product, the \(N=1\) sweep case requires an `empty_product` callback or value.
- Ordinary range queries always use the segment-tree cache. `sweep` separately pre-analyzes `order` and chooses prefix/suffix, two-stack deque, or segment-tree execution.
- The selected strategy is pattern-optimal according to the table in the prompt; this does not claim to solve the unrestricted algebraic circuit-minimization problem for arbitrary orders.

## `WordReduction.jl`

```julia
module WordReduction

using MPSKit: PeriodicVector

export PeriodicWord,
       SweepLog,
       sweep,
       cache_stats,
       cache_storage,
       sweep_log,
       reset_stats!

# ---------------------------------------------------------------------------
# Statistics and logging
# ---------------------------------------------------------------------------

mutable struct CacheCounters
    multiplications::Int
    hits::Int
    misses::Int
    invalidations::Int
end

CacheCounters() = CacheCounters(0, 0, 0, 0)

"""
Summary of one completed sweep.

Fields
------
- `strategy`: `:prefix_suffix`, `:two_stack_deque`, or `:segment_tree`
- `steps`: number of updates
- `turns`: number of direction changes detected in the sequential pattern
- `multiplications`: number of internal `*` calls during the sweep
- `invalidations`: number of valid segment-tree nodes invalidated
- `auxiliary_slots`: peak number of deque value/aggregate slots
"""
struct SweepLog
    strategy::Symbol
    steps::Int
    turns::Int
    multiplications::Int
    invalidations::Int
    auxiliary_slots::Int
end

# ---------------------------------------------------------------------------
# Lazy segment tree
# ---------------------------------------------------------------------------

"""
A lazy ordered-product segment tree.

Leaves are initialized eagerly, but internal products are generated only when
queried. Assignment updates the leaf and marks only its ancestor path invalid.
"""
mutable struct ProductTree{T}
    n::Int
    base::Int

    # `nodes[node]` is meaningful only when `valid[node]` is true.
    nodes::Vector{T}
    valid::BitVector

    # `present[node]` distinguishes real intervals from padding beyond n.
    present::BitVector
end

function ProductTree(data::AbstractVector{T}) where {T}
    n = length(data)
    n > 0 || throw(ArgumentError("PeriodicWord cannot be empty"))

    base = 1
    while base < n
        base <<= 1
    end

    nodes = Vector{T}(undef, 2 * base)
    valid = falses(2 * base)
    present = falses(2 * base)

    @inbounds for i in 1:n
        node = base + i - 1
        nodes[node] = data[i]
        valid[node] = true
        present[node] = true
    end

    if base > 1
        @inbounds for node in (base - 1):-1:1
            present[node] = present[node << 1] |
                            present[(node << 1) + 1]
        end
    end

    return ProductTree{T}(n, base, nodes, valid, present)
end

# ---------------------------------------------------------------------------
# PeriodicWord
# ---------------------------------------------------------------------------

"""
    PeriodicWord(data::PeriodicVector; empty_product=nothing)
    PeriodicWord(data::AbstractVector; empty_product=nothing)

A periodic sequence with cached ordered products.

For lengths `N ≤ 16`, `N` is embedded in the `PeriodicWord` type. For larger
periods, a dynamic-length specialization is used.

`empty_product` is needed only when an empty reduction is requested, principally
for `sweep` on a one-element word. It can be either:

1. A precomputed identity value; or
2. A callable `(w, i) -> identity_for_gap_at_i`.

The callback form is appropriate when the identity depends on tensor spaces.
"""
mutable struct PeriodicWord{
    T,
    N,
    V <: PeriodicVector{T},
    E
} <: AbstractVector{T}
    data::V
    tree::ProductTree{T}
    empty_product::E
    counters::CacheCounters
    logs::Vector{SweepLog}
end

function PeriodicWord(
    data::V;
    empty_product=nothing
) where {T, V <: PeriodicVector{T}}
    n = length(data)
    n > 0 || throw(ArgumentError("PeriodicWord cannot be empty"))

    static_n = n <= 16 ? n : 0

    return PeriodicWord{
        T,
        static_n,
        V,
        typeof(empty_product)
    }(
        data,
        ProductTree(data),
        empty_product,
        CacheCounters(),
        SweepLog[]
    )
end

function PeriodicWord(
    data::AbstractVector{T};
    empty_product=nothing
) where {T}
    return PeriodicWord(
        PeriodicVector(collect(data));
        empty_product=empty_product
    )
end

@inline function Base.length(
    w::PeriodicWord{T, N}
) where {T, N}
    return N == 0 ? length(w.data) : N
end

Base.size(w::PeriodicWord) = (length(w),)
Base.axes(w::PeriodicWord) = (Base.OneTo(length(w)),)
Base.IndexStyle(::Type{<:PeriodicWord}) = IndexLinear()
Base.eltype(::Type{<:PeriodicWord{T}}) where {T} = T
Base.parent(w::PeriodicWord) = w.data

@inline function _canonical_index(w::PeriodicWord, i::Integer)
    return mod1(Int(i), length(w))
end

@inline function Base.getindex(w::PeriodicWord, i::Integer)
    return @inbounds w.data[_canonical_index(w, i)]
end

@inline function _mul!(w::PeriodicWord, a, b)
    w.counters.multiplications += 1
    return a * b
end

"""
Update one leaf and invalidate only its segment-tree ancestor path.

No `*` calls are performed here.
"""
function _set_leaf!(w::PeriodicWord, i::Int, value)
    tree = w.tree
    leaf = tree.base + i - 1

    @inbounds begin
        tree.nodes[leaf] = value
        tree.valid[leaf] = true
    end

    node = leaf >>> 1

    @inbounds while node != 0
        if tree.valid[node]
            tree.valid[node] = false
            w.counters.invalidations += 1
        end
        node = node >>> 1
    end

    return nothing
end

function Base.setindex!(
    w::PeriodicWord,
    value,
    i::Integer
)
    k = _canonical_index(w, i)

    # PeriodicVector performs any required conversion to its element type.
    @inbounds w.data[k] = value

    # Read the converted/stored value back.
    stored = @inbounds w.data[k]
    _set_leaf!(w, k, stored)

    return value
end

# ---------------------------------------------------------------------------
# Lazy tree evaluation
# ---------------------------------------------------------------------------

"""
Return the cached product for a tree node, computing missing descendants in
left-to-right order.
"""
function _node_value!(
    w::PeriodicWord{T},
    node::Int
) where {T}
    tree = w.tree

    @inbounds if tree.valid[node]
        w.counters.hits += 1
        return tree.nodes[node]
    end

    w.counters.misses += 1

    node < tree.base ||
        error("internal error: an invalid segment-tree leaf was encountered")

    left = node << 1
    right = left + 1

    left_present = @inbounds tree.present[left]
    right_present = @inbounds tree.present[right]

    value = if left_present && right_present
        _mul!(
            w,
            _node_value!(w, left),
            _node_value!(w, right)
        )
    elseif left_present
        _node_value!(w, left)
    elseif right_present
        _node_value!(w, right)
    else
        error("internal error: requested an absent segment-tree node")
    end

    @inbounds begin
        tree.nodes[node] = value
        tree.valid[node] = true
    end

    return value
end

@inline _append_right!(w, ::Nothing, value) = Some(value)

@inline function _append_right!(w, aggregate::Some, value)
    return Some(_mul!(w, aggregate.value, value))
end

@inline _prepend_left!(w, ::Nothing, value) = Some(value)

@inline function _prepend_left!(w, aggregate::Some, value)
    return Some(_mul!(w, value, aggregate.value))
end

"""
Compute the product `w[left] * ... * w[right]` for canonical indices satisfying

    1 ≤ left ≤ right ≤ N.
"""
function _linear_reduct(
    w::PeriodicWord,
    left::Int,
    right::Int
)
    1 <= left <= right <= length(w) ||
        throw(BoundsError(w, left:right))

    tree = w.tree

    lnode = tree.base + left - 1
    rnode = tree.base + right - 1

    left_accumulator = nothing
    right_accumulator = nothing

    while lnode <= rnode
        if isodd(lnode)
            value = _node_value!(w, lnode)
            left_accumulator =
                _append_right!(w, left_accumulator, value)
            lnode += 1
        end

        if iseven(rnode)
            value = _node_value!(w, rnode)
            right_accumulator =
                _prepend_left!(w, right_accumulator, value)
            rnode -= 1
        end

        lnode >>>= 1
        rnode >>>= 1
    end

    if left_accumulator === nothing
        right_accumulator === nothing &&
            error("internal error: empty linear reduction")
        return right_accumulator.value
    elseif right_accumulator === nothing
        return left_accumulator.value
    else
        return _mul!(
            w,
            left_accumulator.value,
            right_accumulator.value
        )
    end
end

"""
Compute `count` consecutive elements starting at periodic index `start`.

This function requires `1 ≤ count ≤ N`.
"""
function _periodic_reduct(
    w::PeriodicWord,
    start::Integer,
    count::Int
)
    n = length(w)

    1 <= count <= n ||
        throw(ArgumentError(
            "periodic reductions must contain between 1 and $n elements"
        ))

    first_index = mod1(Int(start), n)
    linear_last = first_index + count - 1

    if linear_last <= n
        return _linear_reduct(w, first_index, linear_last)
    end

    right_part = _linear_reduct(w, first_index, n)
    left_count = linear_last - n
    left_part = _linear_reduct(w, 1, left_count)

    return _mul!(w, right_part, left_part)
end

"""
Determine the number of sites in the forward periodic arc `first:last`.

If `last ≥ first`, ordinary Julia range length is respected and ranges longer
than one full period are rejected. If `last < first`, the endpoints are
interpreted as a one-wrap periodic range.
"""
function _periodic_range_count(
    first_index::Int,
    last_index::Int,
    n::Int
)
    if last_index >= first_index
        count = last_index - first_index + 1

        count <= n ||
            throw(ArgumentError(
                "a periodic range may span at most one period ($n sites)"
            ))

        return count
    end

    return mod(last_index - first_index, n) + 1
end

function Base.getindex(
    w::PeriodicWord,
    range::AbstractUnitRange{I}
) where {I <: Integer}
    first_index = Int(first(range))
    last_index = Int(last(range))

    count = _periodic_range_count(
        first_index,
        last_index,
        length(w)
    )

    return _periodic_reduct(w, first_index, count)
end

# ---------------------------------------------------------------------------
# Empty products
# ---------------------------------------------------------------------------

function _empty_reduct(w::PeriodicWord, i::Int)
    provider = w.empty_product

    provider === nothing &&
        throw(ArgumentError(
            """
            sweep on a one-element PeriodicWord needs an empty product.

            Construct it with, for example:

                PeriodicWord(data; empty_product=identity_value)

            or:

                PeriodicWord(
                    data;
                    empty_product=(w, i) -> identity_for_gap(w, i)
                )
            """
        ))

    if applicable(provider, w, i)
        return provider(w, i)
    end

    return provider
end

@inline function _reduct_except(w::PeriodicWord, i::Int)
    n = length(w)

    if n == 1
        return _empty_reduct(w, i)
    end

    return _periodic_reduct(w, i + 1, n - 1)
end

# ---------------------------------------------------------------------------
# Ordered aggregate deque
# ---------------------------------------------------------------------------

"""
A noncommutative aggregate deque.

The represented sequence is

    reverse(left_values) ++ right_values

where:

- the end of `left_values` is the front of the deque;
- the end of `right_values` is the back of the deque;
- `left_aggregates[end]` is the front-to-back product of the left stack;
- `right_aggregates[end]` is the front-to-back product of the right stack.
"""
mutable struct AggregateDeque{T}
    left_values::Vector{T}
    left_aggregates::Vector{T}
    right_values::Vector{T}
    right_aggregates::Vector{T}
end

AggregateDeque{T}() where {T} = AggregateDeque{T}(
    T[],
    T[],
    T[],
    T[]
)

@inline function _push_front!(
    w::PeriodicWord,
    deque::AggregateDeque{T},
    value::T
) where {T}
    aggregate = if isempty(deque.left_aggregates)
        value
    else
        _mul!(w, value, deque.left_aggregates[end])
    end

    push!(deque.left_values, value)
    push!(deque.left_aggregates, aggregate)

    return nothing
end

@inline function _push_back!(
    w::PeriodicWord,
    deque::AggregateDeque{T},
    value::T
) where {T}
    aggregate = if isempty(deque.right_aggregates)
        value
    else
        _mul!(w, deque.right_aggregates[end], value)
    end

    push!(deque.right_values, value)
    push!(deque.right_aggregates, aggregate)

    return nothing
end

function _transfer_right_to_left!(
    w::PeriodicWord,
    deque::AggregateDeque
)
    while !isempty(deque.right_values)
        value = pop!(deque.right_values)
        pop!(deque.right_aggregates)
        _push_front!(w, deque, value)
    end

    return nothing
end

function _transfer_left_to_right!(
    w::PeriodicWord,
    deque::AggregateDeque
)
    while !isempty(deque.left_values)
        value = pop!(deque.left_values)
        pop!(deque.left_aggregates)
        _push_back!(w, deque, value)
    end

    return nothing
end

function _pop_front!(
    w::PeriodicWord,
    deque::AggregateDeque
)
    if isempty(deque.left_values)
        _transfer_right_to_left!(w, deque)
    end

    isempty(deque.left_values) &&
        throw(ArgumentError("cannot pop from an empty aggregate deque"))

    pop!(deque.left_aggregates)
    return pop!(deque.left_values)
end

function _pop_back!(
    w::PeriodicWord,
    deque::AggregateDeque
)
    if isempty(deque.right_values)
        _transfer_left_to_right!(w, deque)
    end

    isempty(deque.right_values) &&
        throw(ArgumentError("cannot pop from an empty aggregate deque"))

    pop!(deque.right_aggregates)
    return pop!(deque.right_values)
end

function _deque_reduct!(
    w::PeriodicWord,
    deque::AggregateDeque
)
    left_empty = isempty(deque.left_aggregates)
    right_empty = isempty(deque.right_aggregates)

    left_empty && right_empty &&
        throw(ArgumentError("cannot reduce an empty aggregate deque"))

    if left_empty
        return deque.right_aggregates[end]
    elseif right_empty
        return deque.left_aggregates[end]
    else
        return _mul!(
            w,
            deque.left_aggregates[end],
            deque.right_aggregates[end]
        )
    end
end

"""
Construct the complement of `current` in ordered form:

    w[current + 1] * ... * w[current - 1].

For a forward sweep, the initial complement is stored in the left stack so
successive front removals are cheap. For a backward sweep it is stored in the
right stack so successive back removals are cheap.
"""
function _initialize_deque(
    w::PeriodicWord{T},
    current::Int,
    initial_direction::Int
) where {T}
    n = length(w)
    deque = AggregateDeque{T}()

    sizehint!(deque.left_values, n - 1)
    sizehint!(deque.left_aggregates, n - 1)
    sizehint!(deque.right_values, n - 1)
    sizehint!(deque.right_aggregates, n - 1)

    if initial_direction == 1
        # Insert in reverse so the front site ends up on top of the left stack.
        for offset in (n - 1):-1:1
            value = w[current + offset]
            _push_front!(w, deque, value)
        end
    else
        for offset in 1:(n - 1)
            value = w[current + offset]
            _push_back!(w, deque, value)
        end
    end

    return deque
end

"""
Move the current position by one site.

For a forward move `i -> i+1`:

    old complement = [w[i+1], ..., w[i-1]]
    new complement = [w[i+2], ..., w[i-1], new_w[i]]

For a backward move `i -> i-1`:

    old complement = [w[i+1], ..., w[i-1]]
    new complement = [new_w[i], w[i+1], ..., w[i-2]]
"""
function _advance_deque!(
    w::PeriodicWord,
    deque::AggregateDeque,
    new_value,
    direction::Int
)
    if direction == 1
        _pop_front!(w, deque)
        _push_back!(w, deque, new_value)
    elseif direction == -1
        _pop_back!(w, deque)
        _push_front!(w, deque, new_value)
    else
        error("internal error: non-sequential deque advance")
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Sweep-order analysis
# ---------------------------------------------------------------------------

struct SweepPlan
    strategy::Symbol
    turns::Int
    directions::Vector{Int}
end

"""
Return:

- `1` if `b` is the forward periodic neighbor of `a`;
- `-1` if `b` is the backward periodic neighbor of `a`;
- `0` otherwise.
"""
@inline function _step_direction(a::Int, b::Int, n::Int)
    a == b && return 0

    b == mod1(a + 1, n) && return 1
    b == mod1(a - 1, n) && return -1

    return 0
end

function _analyze_order(
    canonical_order::Vector{Int},
    n::Int
)
    number_of_steps = length(canonical_order)

    if number_of_steps <= 1
        strategy = n == 1 ? :segment_tree : :prefix_suffix
        return SweepPlan(strategy, 0, Int[])
    end

    if n == 1
        return SweepPlan(
            :segment_tree,
            0,
            zeros(Int, number_of_steps - 1)
        )
    end

    directions = Vector{Int}(undef, number_of_steps - 1)

    previous_direction = 0
    turns = 0
    arbitrary = false

    @inbounds for step in eachindex(directions)
        direction = _step_direction(
            canonical_order[step],
            canonical_order[step + 1],
            n
        )

        directions[step] = direction

        if direction == 0
            arbitrary = true
            continue
        end

        if previous_direction != 0 &&
           direction != previous_direction
            turns += 1
        end

        previous_direction = direction
    end

    strategy = if arbitrary || turns > 1
        :segment_tree
    elseif turns == 1
        :two_stack_deque
    else
        :prefix_suffix
    end

    return SweepPlan(strategy, turns, directions)
end

# ---------------------------------------------------------------------------
# Sweep implementations
# ---------------------------------------------------------------------------

function _sweep_deque!(
    w::PeriodicWord,
    updaterule,
    order::Vector{Int},
    plan::SweepPlan
)
    n = length(w)
    n > 1 ||
        error("internal error: deque sweep requires at least two sites")

    initial_direction = isempty(plan.directions) ?
                        1 :
                        plan.directions[1]

    deque = _initialize_deque(
        w,
        order[1],
        initial_direction
    )

    @inbounds for step in eachindex(order)
        current = order[step]
complement = _deque_reduct!(w, deque)
        updated_value = updaterule(w, complement, current)

        w[current] = updated_value

        if step < length(order)
            # Retrieve the actual value stored by PeriodicVector, after any
            # element-type conversion performed by setindex!.
            stored_value = w[current]

            _advance_deque!(
                w,
                deque,
                stored_value,
                plan.directions[step]
            )
        end
    end

    return nothing
end

function _sweep_tree!(
    w::PeriodicWord,
    updaterule,
    order::Vector{Int}
)
    # Eager tree preprocessing. This costs N - 1 multiplications when the tree
    # is completely cold, after which an update affects only one root path.
    _node_value!(w, 1)

    @inbounds for current in order
        complement = _reduct_except(w, current)
        w[current] = updaterule(w, complement, current)
    end

    return nothing
end

"""
    sweep(w::PeriodicWord, updaterule, order)

For each `i` in `order`, perform:

```julia
Wᵢ = w[i + 1 : i - 1]
w[i] = updaterule(w, Wᵢ, i)
```

The `updaterule` callable receives the word (for reading additional state), 
the complement product, and the current position. It may close over external
mutable state and update it as a side effect.

The order is scanned before the first complement query.

Strategy selection:

- no direction change: prefix/suffix representation;
- one direction change: aggregate two-stack deque;
- multiple changes or arbitrary jumps: lazy segment tree.

Integer positions in `order` are interpreted periodically.
"""
function sweep(
    w::PeriodicWord,
    updaterule,
    order::AbstractVector{<:Integer}
)
    n = length(w)
    canonical_order = Vector{Int}(undef, length(order))

    @inbounds for i in eachindex(order)
        canonical_order[i] = mod1(Int(order[i]), n)
    end

    if isempty(canonical_order)
        push!(
            w.logs,
            SweepLog(:prefix_suffix, 0, 0, 0, 0, 0)
        )
        return nothing
    end

    plan = _analyze_order(canonical_order, n)

    multiplications_before = w.counters.multiplications
    invalidations_before = w.counters.invalidations

    if plan.strategy == :segment_tree
        _sweep_tree!(w, updaterule, canonical_order)
        auxiliary_slots = 0
    else
        _sweep_deque!(
            w,
            updaterule,
            canonical_order,
            plan
        )

        # N-1 values and N-1 stack aggregates, split between two stacks.
        auxiliary_slots = 2 * (n - 1)
    end

    push!(
        w.logs,
        SweepLog(
            plan.strategy,
            length(canonical_order),
            plan.turns,
            w.counters.multiplications - multiplications_before,
            w.counters.invalidations - invalidations_before,
            auxiliary_slots
        )
    )

    return nothing
end

# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

"""
Return cumulative cache and multiplication counters.
"""
function cache_stats(w::PeriodicWord)
    counters = w.counters

    return (
        multiplications=counters.multiplications,
        cache_hits=counters.hits,
        cache_misses=counters.misses,
        invalidations=counters.invalidations
    )
end

"""
Return current persistent cache-storage information.

Temporary sweep deque storage is recorded in each `SweepLog`.
"""
function cache_storage(w::PeriodicWord)
    tree = w.tree

    return (
        tree_slots=length(tree.nodes),
        present_tree_nodes=count(identity, tree.present),
        valid_tree_nodes=count(identity, tree.valid),
        sweep_records=length(w.logs)
    )
end

"""
Return a copy of the accumulated sweep log.
"""
sweep_log(w::PeriodicWord) = copy(w.logs)

"""
    reset_stats!(w; clear_log=false)

Reset cumulative counters. Cached products remain intact.
"""
function reset_stats!(
    w::PeriodicWord;
    clear_log::Bool=false
)
    counters = w.counters
    counters.multiplications = 0
    counters.hits = 0
    counters.misses = 0
    counters.invalidations = 0

    clear_log && empty!(w.logs)

    return w
end

end # module WordReduction
```

## Tests

These tests use strings because string concatenation through `*` is associative and noncommutative, making order errors visible.

```julia
using Test
using MPSKit: PeriodicVector

include("../src/WordReduction.jl")
using .WordReduction

function naive_product(values)
    isempty(values) &&
        throw(ArgumentError("naive_product needs a nonempty vector"))

    result = values[1]

    for i in 2:length(values)
        result = result * values[i]
    end

    return result
end

function naive_complement(values, i)
    n = length(values)

    return naive_product([
        values[mod1(i + offset, n)]
        for offset in 1:(n - 1)
    ])
end

function naive_sweep!(values, updaterule, order)
    n = length(values)

    for raw_i in order
        i = mod1(raw_i, n)
        values[i] = updaterule(naive_complement(values, i))
    end

    return nothing
end

@testset "Periodic range products" begin
    w = PeriodicWord(["a", "b", "c", "d", "e"])

    @test w[1] == "a"
    @test w[0] == "e"
    @test w[6] == "a"

    @test w[2:4] == "bcd"
    @test w[4:2] == "deab"
    @test w[2:1] == "bcdea"
    @test w[1:5] == "abcde"

    @test_throws ArgumentError w[1:6]
end

@testset "Tree cache reuse" begin
    w = PeriodicWord([
        "a", "b", "c", "d",
        "e", "f", "g", "h"
    ])

    reset_stats!(w)

    @test w[1:4] == "abcd"
    first_stats = cache_stats(w)

    # Exact cached segment: no new tree construction is needed.
    @test w[1:4] == "abcd"
    second_stats = cache_stats(w)

    @test second_stats.cache_hits > first_stats.cache_hits
    @test second_stats.cache_misses == first_stats.cache_misses
    @test second_stats.multiplications == first_stats.multiplications

    # An overlapping query reuses cached subsegments.
    @test w[2:4] == "bcd"
    third_stats = cache_stats(w)

    @test third_stats.cache_hits > second_stats.cache_hits
end

@testset "Point invalidation" begin
    w = PeriodicWord([
        "a", "b", "c", "d",
        "e", "f", "g", "h"
    ])

    # Materialize the complete tree.
    @test w[1:8] == "abcdefgh"

    before = cache_stats(w)
    w[3] = "X"
    after = cache_stats(w)

    # For N = 8, only the three internal ancestors of leaf 3 are invalidated.
    @test after.invalidations - before.invalidations <= 3
    @test w[1:4] == "abXd"
end

@testset "Forward sweep" begin
    initial = ["a", "b", "c", "d", "e"]
    expected = copy(initial)

    updaterule = complement -> "(" * complement * ")"
    order = collect(1:length(initial))

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :prefix_suffix
end

@testset "Backward sweep" begin
    initial = ["a", "b", "c", "d", "e"]
    expected = copy(initial)

    updaterule = complement -> "[" * complement * "]"
    order = collect(length(initial):-1:1)

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :prefix_suffix
end

@testset "One-turn sweep" begin
    n = 5
    initial = ["a", "b", "c", "d", "e"]
    expected = copy(initial)

    updaterule = complement -> "{" * complement * "}"
    order = vcat(collect(1:n), collect((n - 1):-1:2))

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :two_stack_deque
    @test last(sweep_log(w)).turns == 1
end

@testset "Multiple turns use tree" begin
    initial = ["a", "b", "c", "d", "e"]
    expected = copy(initial)

    updaterule = complement -> "<" * complement * ">"
    order = [1, 2, 1, 2, 1]

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :segment_tree
end

@testset "Arbitrary order uses tree" begin
    initial = ["a", "b", "c", "d", "e"]
    expected = copy(initial)

    updaterule = complement -> "X" * complement
    order = [1, 3, 5, 2, 4]

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :segment_tree
end

@testset "Periodic order indices" begin
    initial = ["a", "b", "c", "d"]
    expected = copy(initial)

    updaterule = complement -> "(" * complement * ")"

    # Canonically this is 1, 2, 3, 4.
    order = [5, 6, 7, 8]

    naive_sweep!(expected, updaterule, order)

    w = PeriodicWord(initial)
    sweep(w, updaterule, order)

    @test collect(w) == expected
    @test last(sweep_log(w)).strategy == :prefix_suffix
end

@testset "One-element identity provider" begin
    w = PeriodicWord(
        ["a"];
        empty_product=(word, i) -> ""
    )

    sweep(w, _ -> "updated", [1])

    @test w[1] == "updated"
    @test last(sweep_log(w)).strategy == :segment_tree
end
```

## Complexity

Let \(N=\operatorname{length}(w)\).

| Operation | Multiplications | Invalidations |
|---|---:|---:|
| `w[i]` | \(0\) | \(0\) |
| `w[i] = a` | \(0\) | \(O(\log N)\) |
| Cold tree construction | \(N-1\) | \(0\) |
| Cached linear range query | \(O(\log N)\) joins, often fewer | \(0\) |
| Sequential sweep | \(O(N)\) preprocessing, \(O(1)\) amortized/update | \(O(\log N)\) metadata/update |
| One-turn sweep | \(O(N)\) preprocessing, \(O(1)\) amortized/update | \(O(\log N)\) metadata/update |
| Arbitrary sweep | \(O(N)\) preprocessing, \(O(\log N)\)/update | \(O(\log N)\)/update |

The `setindex!` invalidation cost is only metadata traversal: it performs no tensor multiplication. During optimized sweeps, the segment tree remains lazy while the deque maintains the products needed by the sweep. A later ordinary range query reconstructs only invalid tree paths.

For space-dependent tensor identities, construction would look conceptually like:

```julia
w = PeriodicWord(
    tensors;
    empty_product=(w, i) -> identity_for_contraction_gap(w, i)
)
```

This keeps `WordReduction` independent of TensorKit while allowing the caller to construct the correct identity from tensor-domain and codomain spaces.
