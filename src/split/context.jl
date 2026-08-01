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

function SplitContext(mpsA::PeriodicMPS{<:AbstractTensorMap{E, S, 3, 1}}, mpsB::PeriodicMPS{<:AbstractTensorMap{E, S, 2, 1}}) where {E, S}
    braketBB = LoopTransferMatrix(mpsB, mpsB)
    braketAB = LoopTransferMatrix(mpsA, mpsB)
    braketAA = LoopTransferMatrix(mpsA, mpsA)
    return SplitContext(mpsA, mpsB, braketAA, braketBB, braketAB,
                        build_cache(braketBB, 1),
                        build_cache(braketAB, 1),
                        expectation(braketAA),
                        SplitLog())
end

"""
    update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)

One site's update: environment (O(1)) → update rule → write back the new
B tensor and its transfer matrix. Rebuilds AB coupling when appropriate.
"""
function update!(ctx::SplitContext, solve::AbstractSolve, st::SweepState)
    env = environment(ctx.cacheBB)            # cut == st.site
    new = _update_rule(ctx, solve, env)         # TODO: 物理规则
    ctx.mpsB[st.site] = new
    ctx.braketBB[st.site] = site_transfer_matrix(new)
    if _should_update_AB(st)
        pa = pa_from_pb(st.site)
        ctx.braketAB[pa] = site_transfer_matrix(ctx.mpsA[pa],
                                                ctx.mpsB[2 * pa - 1], ctx.mpsB[2 * pa])
    end
    return new
end

# A site index from B site index: A[pa] couples to B[2pa-1], B[2pa]
pa_from_pb(site::Int) = (site + 1) ÷ 2

# 是否应该重建 AB 耦合 —— 取决于 sweep 方向
_should_update_AB(st::SweepState) =
    st.dir == 1 && iseven(st.site) || st.dir == -1 && isodd(st.site)

# 初始 1-site 近似的构造 —— 物理内容，后续实现
_initial_approximation(mps, alg) =
    error("TODO: initial 1-site approximation not implemented yet")

# 每 site 的更新规则 —— 物理内容，后续实现
_update_rule(ctx, alg, env) =
    error("TODO: site update rule not implemented yet")

# 工厂：从 MPS 构造迭代器（sweep.jl 只依赖 N，不依赖 PeriodicMPS）
ZigzagSweep(mps::PeriodicMPS; maximal_sweep::Int)   = ZigzagSweep(length(mps); maximal_sweep)
CircularSweep(mps::PeriodicMPS; maximal_sweep::Int) = CircularSweep(length(mps); maximal_sweep)

# ------------------------------------------------------------------------------
# Driver — the sweep scheme iterates itself
# ------------------------------------------------------------------------------

function _split(mps::PeriodicMPS, alg::LoopSplit)
    ctx = SplitContext(mps, _initial_approximation(mps, alg))
    for st in alg.sweep
        update!(ctx, alg.solve, st)
        advance!(ctx.cacheBB, ctx.braketBB, st)
        if _should_update_AB(st)
            advance!(ctx.cacheAB, ctx.braketAB, st)
        end
    end
    return ctx.mpsB, ctx.log
end
