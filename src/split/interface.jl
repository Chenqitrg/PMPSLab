# ------------------------------------------------------------------------------
# 2-site → 1-site splitting — inverse of pair_contraction
# ------------------------------------------------------------------------------

abstract type AbstractAnsatz <: AbstractAlgorithmConfig end   # initial ansatz for the 2-site → 1-site splitting
abstract type AbstractSolve <: AbstractAlgorithmConfig end   # 每 site 更新规则
abstract type AbstractSweep <: AbstractAlgorithmConfig end   # 扫描方案：Zigzag, Circular
abstract type AbstractALSSplit <: AbstractAlgorithmConfig end  # alternating least-squares split

"""
    SplitEnvironmentPolicy(; norm=DenseTransferPolicy(), mixed=DenseTransferPolicy())

Choose the transfer representation independently for the split normal
environment `⟨B|B⟩` and mixed environment `⟨B|A⟩`. Keeping these policies
separate permits, for example, an exact normal equation with a compressed
right-hand-side environment.
"""
@kwdef struct SplitEnvironmentPolicy{
        N<:AbstractTransferPolicy,
        M<:AbstractTransferPolicy,
    } <: AbstractAlgorithmConfig
    norm::N = DenseTransferPolicy()
    mixed::M = DenseTransferPolicy()
end

"""
    ALSSplit(;
        ansatz,
        sweep,
        solve,
        environments=SplitEnvironmentPolicy(),
        stopping=NoStopping(),
    )
Configure an alternating least-squares (ALS) periodic 2-to-1 split by composing
its initial ansatz, sweep order, local solver, and normal/mixed transfer
representations. An optional `stopping` criterion may end execution at a
complete sweep boundary before the sweep iterator is exhausted.
"""
@kwdef struct ALSSplit{
        A<:AbstractAnsatz,
        S<:AbstractSweep,
        V<:AbstractSolve,
        E<:SplitEnvironmentPolicy,
        C<:AbstractStoppingCriterion,
    } <: AbstractALSSplit
    ansatz::A
    sweep::S
    solve::V
    environments::E = SplitEnvironmentPolicy()
    stopping::C = NoStopping()
end

# split(mps, alg) defined in run.jl — drives SplitRun to completion
