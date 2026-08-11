# ------------------------------------------------------------------------------
# 2-site → 1-site splitting — inverse of pair_contraction
# ------------------------------------------------------------------------------

abstract type AbstractAnsatz <: AbstractAlgorithmConfig end   # initial ansatz for the 2-site → 1-site splitting
abstract type AbstractSolve <: AbstractAlgorithmConfig end   # 每 site 更新规则
abstract type AbstractSweep <: AbstractAlgorithmConfig end   # 扫描方案：Zigzag, Circular
abstract type AbstractPeriodicSplit <: AbstractAlgorithmConfig end  # Split = sweep + solve 的组合

"""
    SplitEnvironmentPolicy(; norm=DenseTransferPolicy(), mixed=DenseTransferPolicy())

Choose the transfer representation independently for the split normal
environment `⟨B|B⟩` and mixed environment `⟨B|A⟩`. Keeping these policies
separate permits, for example, an exact normal equation with a compressed
right-hand-side environment.
"""
struct SplitEnvironmentPolicy{
        N<:AbstractTransferPolicy,
        M<:AbstractTransferPolicy,
    } <: AbstractAlgorithmConfig
    norm::N
    mixed::M
end

function SplitEnvironmentPolicy(;
        norm::AbstractTransferPolicy = DenseTransferPolicy(),
        mixed::AbstractTransferPolicy = DenseTransferPolicy(),
    )
    return SplitEnvironmentPolicy(norm, mixed)
end

SplitEnvironmentPolicy(policy::AbstractTransferPolicy) =
    SplitEnvironmentPolicy(policy, policy)

"""
    PeriodicSplit(ansatz, sweep, solve; environments=SplitEnvironmentPolicy())
    PeriodicSplit(ansatz, sweep, solve, environments)

Compose the initial ansatz, sweep order, local solver, and normal/mixed transfer
representations for a periodic 2-to-1 split.
"""
struct PeriodicSplit{
        A<:AbstractAnsatz,
        S<:AbstractSweep,
        V<:AbstractSolve,
        E<:SplitEnvironmentPolicy,
    } <: AbstractPeriodicSplit
    ansatz::A
    sweep::S
    solve::V
    environments::E
end

function PeriodicSplit(
        ansatz::AbstractAnsatz,
        sweep::AbstractSweep,
        solve::AbstractSolve;
        environments::SplitEnvironmentPolicy = SplitEnvironmentPolicy(),
    )
    return PeriodicSplit(ansatz, sweep, solve, environments)
end

# split(mps, alg) defined in run.jl — drives SplitRun to completion
