# ------------------------------------------------------------------------------
# SplitContext — state of one 2-site → 1-site split run
#
# The context is the single mutable state object; `update!` performs one site's
# update in place. The sweep scheme (ZigzagSweep / CircularSweep) supplies the
# visit order; the update rule itself is the physics, filled in per algorithm.
# ------------------------------------------------------------------------------

# A site (2 physical legs) maps to two B sites (1 physical leg each):
#   pa = (pb + 1) ÷ 2, and A[pa] couples to B[2pa-1], B[2pa].
mutable struct SplitContext{E, S} <: AlgorithmContext{E, S}
    mpsA            :: PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}      # 2-site mps（N/2 个 site，只读）
    mpsB            :: PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}      # 1-site mps（N 个 site，update! 就地修改）
    braketAA          :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 2-site transfer matrix 环
    braketBB           :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 1-site transfer matrix 环
    braketAB           :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 2-site → 1-site transfer matrix 环
    cacheBB         :: RingCache{<:AbstractTensorMap{E, S, 2, 2}}       # braketBB 的滑动缓存
    cacheAB         :: RingCache{<:AbstractTensorMap{E, S, 2, 2}}        # braketAB 的滑动缓存
    expectationAA  :: E
end

site(ctx::SplitContext) = ctx.cacheBB.site
this_B(ctx::SplitContext) = ctx.mpsB[site(ctx)]
other_B(ctx::SplitContext) = iseven(site(ctx)) ? ctx.mpsB[site(ctx) - 1] : ctx.mpsB[site(ctx) + 1]
pa_from_pb(pb::Int) = (pb + 1) ÷ 2
this_A(ctx::SplitContext) = ctx.mpsA[pa_from_pb(site(ctx))]

function SplitContext(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    braketBB = LoopTransferMatrix(mpsB, mpsB)
    braketAB = LoopTransferMatrix(mpsB, mpsA)
    braketAA = LoopTransferMatrix(mpsA, mpsA)
    return SplitContext(mpsA, mpsB, braketAA, braketBB, braketAB,
                        build_cache(braketBB, 1),
                        build_cache(braketAB, 1),
                        expectation(braketAA))
end

function solvenew(ctx::SplitContext, alg::AbstractSolve)
    N = transpose(environment(ctx.cacheBB), ((3, 1), (4, 2)))   # N = E₁ * ... * E_N
    AB_env = environment(ctx.cacheAB)
    A = this_A(ctx)
    oB = other_B(ctx)'
    if isodd(site(ctx))
        @plansor W[-1; -2 -3] := AB_env[1 2; 3 -2] * A[3 -1 4; 1] * oB[2; -3 4]
    else
        @plansor W[-1; -2 -3] := AB_env[1 -3; 2 4] * A[2 3 -1; 1] * oB[-2; 4 3]
    end
    x0 = transpose(this_B(ctx), ((2,), (1, 3)))  # B → (P ← V_left, V_right) for right-division
    x1 = rdiv(alg, x0, N, W)

    # 每 site 成本 ‖ψ_A − ψ_B‖² 的局部环境近似（最优解处 tr(W * x1') = ⟨ψ_B|ψ_B⟩）
    cost = real(ctx.expectationAA - conj(tr(W * x1')))

    return transpose(x1, ((2, 1), (3,))), cost    # back to (V_left ⊗ P ← V_right)
end

"""
    update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)

One site's update + cache advance. Environment (O(1)) → update rule →
write back the new B tensor and transfer matrices → slide both caches.
"""
function update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)
    site(ctx) != st.site && throw(ArgumentError("sweep state site $(st.site) does not match context site $(site(ctx))"))

    new, cost = solvenew(ctx, solve)
    ctx.mpsB[st.site] = new
    ctx.braketBB[st.site] = site_transfer_matrix(new)
    advance!(ctx.cacheBB, ctx.braketBB, st)
    if (st.dir == 1 && iseven(st.site)) || (st.dir == -1 && isodd(st.site))
        pa = pa_from_pb(st.site)
        ctx.braketAB[pa] = site_transfer_matrix(ctx.mpsA[pa],
                                                ctx.mpsB[2 * pa - 1], ctx.mpsB[2 * pa])
        advance!(ctx.cacheAB, ctx.braketAB, st)
    end
    return new, cost
end

# 工厂：从 MPS 构造迭代器（sweep.jl 只依赖 N，不依赖 PeriodicMPS）
ZigzagSweep(mps::PeriodicMPS; maximal_sweep::Int)   = ZigzagSweep(length(mps); maximal_sweep)
CircularSweep(mps::PeriodicMPS; maximal_sweep::Int) = CircularSweep(length(mps); maximal_sweep)

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

"""
    split(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}, alg::AbstractLoopSplit)

Given a periodic MPS with 2 physical legs per site (3 codim legs), produce a
periodic MPS with 1 physical leg per site (2 codim legs) that best approximates
the loop state. Returns `(mps_1site, log)` where `log` is a [`SplitLog`](@ref)
(= `SweepLog{SiteLog{SweepState}}`) — one record `(st, cost, time)` per
update, in visit order; per-sweep grouping follows from `st.n`.
"""
function split(mps::PeriodicMPS, alg::LoopSplit)
    ctx = SplitContext(mps, initial_approximation(mps, alg.ansatz))
    log = SplitLog()
    for st in alg.sweep
        t0 = time()
        _, cost = update!(ctx, alg.solve, st)
        push!(log.site_logs, SiteLog(st, cost, time() - t0))
    end
    return ctx.mpsB, log
end
