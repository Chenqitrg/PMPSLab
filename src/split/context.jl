# ------------------------------------------------------------------------------
# SplitContext — state of one 2-site → 1-site split run
#
# The context owns mutable numerical working state but no sweep position.
# `update!` performs one site's update in place from the supplied `SweepState`.
# The sweep scheme supplies the visit order; the update rule itself is the
# physics, filled in per algorithm.
# ------------------------------------------------------------------------------

# A site (2 physical legs) maps to two B sites (1 physical leg each):
#   pa = (pb + 1) ÷ 2, and A[pa] couples to B[2pa-1], B[2pa].
struct SplitContext{
        E, S,
        MA<:PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, MB,
        TAA, TBB, TAB, CBB, CAB
    } <: AlgorithmContext{E,S}
    mpsA::MA
    mpsB::MB
    braketAA::TAA
    braketBB::TBB
    braketAB::TAB
    cacheBB::CBB
    cacheAB::CAB
    expectationAA::E
end

pa_from_pb(pb::Int) = cld(pb, 2)
this_B(ctx::SplitContext, pb::Int) = ctx.mpsB[pb]
other_B(ctx::SplitContext, pb::Int) = iseven(pb) ? ctx.mpsB[pb - 1] : ctx.mpsB[pb + 1]
this_A(ctx::SplitContext, pb::Int) = ctx.mpsA[pa_from_pb(pb)]

function SplitContext(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    braketBB = PeriodicTransferMatrix(mpsB, mpsB)
    braketAB = PeriodicTransferMatrix(mpsB, mpsA)
    braketAA = PeriodicTransferMatrix(mpsA, mpsA)
    return SplitContext(mpsA, mpsB, braketAA, braketBB, braketAB,
                        build_cache(braketBB),
                        build_cache(braketAB),
                        expectation(braketAA))
end

function solvenew(ctx::SplitContext, alg::AbstractSolve, st::SweepState)
    pb = st.site
    1 <= pb <= length(ctx.mpsB) || throw(BoundsError(ctx.mpsB, pb))
    pa = pa_from_pb(pb)
    N = transpose(environment(ctx.cacheBB, pb), ((3, 1), (4, 2)))   # N = E₁ * ... * E_N
    AB_env = environment(ctx.cacheAB, pa)
    A = this_A(ctx, pb)
    oB = other_B(ctx, pb)'
    if isodd(pb)
        @plansor W[-1; -2 -3] := AB_env[1 2; 3 -2] * A[3 -1 4; 1] * oB[2; -3 4]
    else
        @plansor W[-1; -2 -3] := AB_env[1 -3; 2 4] * A[2 3 -1; 1] * oB[-2; 4 3]
    end
    x0 = transpose(this_B(ctx, pb), ((2,), (1, 3)))  # B → (P ← V_left, V_right) for right-division
    x1 = rdiv(alg, x0, N, W)

    # 每 site 成本 ‖ψ_A − ψ_B‖² 的局部环境近似（最优解处 tr(W * x1') = ⟨ψ_B|ψ_B⟩）
    cost = real(ctx.expectationAA - conj(tr(W * x1')))

    return transpose(x1, ((2, 1), (3,))), cost    # back to (V_left ⊗ P ← V_right)
end

"""
    update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)

One site's update + directional cache refresh. Environment (O(1)) → update
rule → write back the new B tensor and transfer matrices → refresh the affected
prefix or suffix entries.
"""
function update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)
    new, cost = solvenew(ctx, solve, st)
    ctx.mpsB[st.site] = new
    ctx.braketBB[st.site] = site_transfer_matrix(new)
    advance!(ctx.cacheBB, ctx.braketBB, st)
    if (st.dir == 1 && iseven(st.site)) || (st.dir == -1 && isodd(st.site))
        pa = pa_from_pb(st.site)
        ctx.braketAB[pa] = site_transfer_matrix(
            ctx.mpsB[2 * pa - 1], ctx.mpsB[2 * pa], ctx.mpsA[pa]
        )
        advance!(ctx.cacheAB, ctx.braketAB, SweepState(pa, st.dir, st.n))
    end
    return new, cost
end

# 工厂：从 MPS 构造迭代器（sweep.jl 只依赖 N，不依赖 PeriodicMPS）
ZigzagSweep(mps::PeriodicMPS; maximal_sweep::Int)   = ZigzagSweep(length(mps); maximal_sweep)
CircularSweep(mps::PeriodicMPS; maximal_sweep::Int) = CircularSweep(length(mps); maximal_sweep)
