# ------------------------------------------------------------------------------
# SplitContext — state of one 2-site → 1-site split run
#
# The context owns mutable numerical working state but no sweep position.
# `update!` performs one site's update in place from the supplied `SweepState`.
# The sweep scheme supplies the visit order. Caches track which derived products
# remain valid and reject unsupported update/query sequences without performing
# implicit tensor contractions.
# ------------------------------------------------------------------------------

# A site (2 physical legs) maps to two B sites (1 physical leg each):
#   pa = (pb + 1) ÷ 2, and A[pa] couples to B[2pa-1], B[2pa].
struct SplitContext{
        E, S,
        MA<:PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, MB,
        CBB, CAB
    } <: AlgorithmContext{E,S}
    mpsA::MA
    mpsB::MB
    cacheBB::CBB
    cacheAB::CAB
    expectationAA::E
end

pa_from_pb(pb::Int) = cld(pb, 2)
this_B(ctx::SplitContext, pb::Int) = ctx.mpsB[pb]
other_B(ctx::SplitContext, pb::Int) = iseven(pb) ? ctx.mpsB[pb - 1] : ctx.mpsB[pb + 1]
this_A(ctx::SplitContext, pb::Int) = ctx.mpsA[pa_from_pb(pb)]

function SplitContext(
        mpsA::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}},
        mpsB::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    cacheBB = build_cache(mpsB, mpsB)
    cacheAB = build_cache(mpsB, mpsA)
    expectationAA = expectation(mpsA, mpsA)
    return SplitContext(
        mpsA,
        mpsB,
        cacheBB,
        cacheAB,
        expectationAA,
    )
end

function solvenew(ctx::SplitContext, alg::AbstractSolve, st::SweepState)
    pb = st.site
    1 <= pb <= length(ctx.mpsB) || throw(BoundsError(ctx.mpsB, pb))
    pa = pa_from_pb(pb)
    BB_env = environment(ctx.cacheBB, pb)
    AB_env = environment(ctx.cacheAB, pa)
    A = this_A(ctx, pb)
    oB = other_B(ctx, pb)'
    W = _split_rhs(AB_env, A, oB, pb)
    x0 = transpose(this_B(ctx, pb), ((2,), (1, 3)))  # B → (P ← V_left, V_right) for right-division
    x1 = _solve_split_norm(alg, x0, BB_env, W)

    # 每 site 成本 ‖ψ_A − ψ_B‖² 的局部环境近似（最优解处 tr(W * x1') = ⟨ψ_B|ψ_B⟩）
    cost = real(ctx.expectationAA - conj(tr(W * x1')))

    return transpose(x1, ((2, 1), (3,))), cost    # back to (V_left ⊗ P ← V_right)
end

_pair_completed(st::SweepState{Forward}) = iseven(st.site)
_pair_completed(st::SweepState{Backward}) = isodd(st.site)

"""
    update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)

One site's update + validated directional cache refresh. An incomplete B pair
only invalidates its mixed AB factor; completing the pair constructs that factor
once and advances the AB cache. Unsupported visit orders fail with
`StaleCacheError` instead of triggering extra contractions or returning stale
environments.
"""
function update!(
        ctx::SplitContext,
        solve::AbstractSolve,
        st::SweepState{D},
    ) where {D}
    new, cost = solvenew(ctx, solve, st)
    ctx.mpsB[st.site] = new

    if _requires_rebuild(ctx.cacheBB, st)
        if D === Forward
            rebuild_right!(ctx.cacheBB, ctx.mpsB, ctx.mpsB)
            rebuild_right!(ctx.cacheAB, ctx.mpsB, ctx.mpsA)
        else
            rebuild_left!(ctx.cacheBB, ctx.mpsB, ctx.mpsB)
            rebuild_left!(ctx.cacheAB, ctx.mpsB, ctx.mpsA)
        end
        return new, cost
    end

    bb_transfer = transfer_at(ctx.mpsB, ctx.mpsB, st.site)
    advance!(ctx.cacheBB, bb_transfer, st)

    pa = pa_from_pb(st.site)
    if _pair_completed(st)
        pair_state = SweepState{D}(pa, st.n)
        ab_transfer = transfer_at(ctx.mpsB, ctx.mpsA, pa)
        advance!(ctx.cacheAB, ab_transfer, pair_state)
    else
        # Avoid constructing the mixed transfer until both B sites are current.
        invalidate!(ctx.cacheAB, pa)
    end
    return new, cost
end

# 工厂：从 MPS 构造迭代器（sweep.jl 只依赖 N，不依赖 PeriodicMPS）
ZigzagSweep(mps::PeriodicMPS; maximal_sweep::Int)   = ZigzagSweep(length(mps); maximal_sweep)
CircularSweep(mps::PeriodicMPS; maximal_sweep::Int) = CircularSweep(length(mps); maximal_sweep)
