# ------------------------------------------------------------------------------
# SplitContext — state of one 2-site → 1-site split run
#
# The context is the single mutable state object; `update!` performs one site's
# update in place. The sweep scheme (ZigzagSweep / CircularSweep) supplies the
# visit order; the update rule itself is the physics, filled in per algorithm.
# ------------------------------------------------------------------------------

# A site (2 physical legs) maps to two B sites (1 physical leg each):
#   pa = (pb + 1) ÷ 2, and A[pa] couples to B[2pa-1], B[2pa].
mutable struct SplitContext{E, S}
    mpsA            :: PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}      # 2-site mps（N/2 个 site，只读）
    mpsB            :: PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}      # 1-site mps（N 个 site，update! 就地修改）
    braketAA          :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 2-site transfer matrix 环
    braketBB           :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 1-site transfer matrix 环
    braketAB           :: LoopTransferMatrix{<:AbstractTensorMap{E, S, 2, 2}}  # 2-site → 1-site transfer matrix 环
    cacheBB         :: RingCache{<:AbstractTensorMap{E, S, 2, 2}}       # braketBB 的滑动缓存
    cacheAB         :: RingCache{<:AbstractTensorMap{E, S, 2, 2}}        # braketAB 的滑动缓存
    expectationAA  :: Float64
    log             :: SplitLog
end

site(ctx::SplitContext) = ctx.cacheBB.site
this_B(ctx::SplitContext) = ctx.mpsB[site(ctx)]
pa_from_pb(pb::Int) = (pb + 1) ÷ 2
this_A(ctx::SplitContext) = ctx.mpsA[pa_from_pb(site(ctx))]

function SplitContext(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    braketBB = LoopTransferMatrix(mpsB, mpsB)
    braketAB = LoopTransferMatrix(mpsA, mpsB)
    braketAA = LoopTransferMatrix(mpsA, mpsA)
    unit = id(E, codomain(braketBB[1]))
    return SplitContext(mpsA, mpsB, braketAA, braketBB, braketAB,
                        build_cache(braketBB, 1, unit),
                        build_cache(braketAB, 1, unit),
                        expectation(braketAA),
                        SplitLog())
end

function construct_equation(ctx::SplitContext)
    N = transpose(environment(ctx.cacheBB), ((1, 2), (3, 4)))   # N = E₁ * ... * E_N
    # TODO: build W from ABenv and this_A, with even/odd parity
    W = zero(this_B(ctx))
    return this_B(ctx), N, W
end

function _update_rule(ctx::SplitContext, alg::LUSolve)
    x, N, W = construct_equation(ctx)
    return solve(alg, x, N, W)
end

"""
    update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)

One site's update + cache advance. Environment (O(1)) → update rule →
write back the new B tensor and transfer matrices → slide both caches.
"""
function update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)
    this_B(ctx) != st.site && throw(ArgumentError("sweep state site $(st.site) does not match context site $(this_B(ctx))"))

    new = _update_rule(ctx, solve)
    ctx.mpsB[st.site] = new
    ctx.braketBB[st.site] = site_transfer_matrix(new)
    advance!(ctx.cacheBB, ctx.braketBB, st)
    if (st.dir == 1 && iseven(st.site)) || (st.dir == -1 && isodd(st.site))
        pa = pa_from_pb(st.site)
        ctx.braketAB[pa] = site_transfer_matrix(ctx.mpsA[pa],
                                                ctx.mpsB[2 * pa - 1], ctx.mpsB[2 * pa])
        advance!(ctx.cacheAB, ctx.braketAB, st)
    end
    return new
end

# 初始 1-site 近似的构造 —— 物理内容，后续实现
_initial_approximation(mps, alg) =
    error("TODO: initial 1-site approximation not implemented yet")

# 每 site 的更新规则 —— 物理内容，后续实现


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
the loop state. Returns `(mps_1site, log)`.
"""
function split(mps::PeriodicMPS, alg::LoopSplit)
    ctx = SplitContext(mps, _initial_approximation(mps, alg))
    for st in alg.sweep
        update!(ctx, alg.solve, st)
    end
    return ctx.mpsB, ctx.log
end
