# ------------------------------------------------------------------------------
# SplitContext — state of one 2-site → 1-site split run
#
# The context is the single mutable state object; `update!` performs one site's
# update in place. The sweep scheme (ZigzagSweep / CircularSweep) supplies the
# visit order; the update rule itself is the physics, filled in per algorithm.
# ------------------------------------------------------------------------------

mutable struct SplitContext
    mps_original :: PeriodicMPS      # 原始 2-site mps（只读参考）
    mps          :: PeriodicMPS      # 当前 1-site 近似，update! 就地修改
    ltm          :: LoopTransferMatrix  # mps 的转移矩阵环（cache 的元素源）
    cache        :: RingCache        # 滑动缓存，site 字段即当前位置
    log          :: SplitLog
end

function SplitContext(mps_original::PeriodicMPS, mps::PeriodicMPS, alg::AbstractSweep)
    ltm = LoopTransferMatrix(mps)
    return SplitContext(mps_original, mps, ltm, build_cache(ltm, 1),
                                SplitLog())
end

"""
    update!(ctx::SplitContext, alg::AbstractSweep, site::Int)

One site's update: environment at the current cut (O(1)) → update rule →
write back the new tensor and its transfer matrix. Returns the new tensor.
"""
function update!(ctx::SplitContext, alg::AbstractSweep, site::Int)
    env = environment(ctx.cache)              # cut == site
    new = _update_rule(ctx, alg, env)         # TODO: 物理规则（按算法类型实现）
    ctx.mps[site] = new
    ctx.ltm.matrices[site] = site_transfer_matrix(new)
    return new
end

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

function _split(mps::PeriodicMPS, alg::AbstractSweep)
    ctx = SplitContext(mps, _initial_approximation(mps, alg), alg)
    for (site, dir) in alg
        update!(ctx, alg, site)
        advance!(ctx.cache, ctx.ltm, ctx.ltm[site], dir)
    end
    return ctx.mps, ctx.log
end
