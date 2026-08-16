# ------------------------------------------------------------------------------
# Two-physical-leg → paired single-physical-leg splitting, inverse of
# `contract_pairs`.
# ------------------------------------------------------------------------------

abstract type AbstractAnsatz <: AbstractAlgorithmConfig end
abstract type AbstractSolve <: AbstractAlgorithmConfig end
abstract type AbstractSweep <: AbstractAlgorithmConfig end
abstract type AbstractALSSplit <: AbstractAlgorithmConfig end

"""
    SplitEnvironmentPolicy(; normal=DenseTransferPolicy(), mixed=DenseTransferPolicy())

Choose the transfer representation independently for the split normal
environment `⟨approximation|approximation⟩` and mixed environment
`⟨approximation|target⟩`. Keeping these policies
separate permits, for example, an exact normal equation with a compressed
right-hand-side environment.
"""
@kwdef struct SplitEnvironmentPolicy{
        N<:AbstractTransferPolicy,
        M<:AbstractTransferPolicy,
    } <: AbstractAlgorithmConfig
    normal::N = DenseTransferPolicy()
    mixed::M = DenseTransferPolicy()
end

"""
    ALSSplit(;
        ansatz,
        sweep,
        solver,
        environments=SplitEnvironmentPolicy(),
        stopping=NoStopping(),
    )
Configure an alternating least-squares (ALS) split from two-physical-leg target
tensors to paired single-physical-leg approximation tensors. It composes the
initial ansatz, sweep order, local solver, and normal/mixed transfer
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
    solver::V
    environments::E = SplitEnvironmentPolicy()
    stopping::C = NoStopping()
end

# `split(target, algorithm)` is defined in run.jl and drives `SplitRun` to
# completion.
