# ------------------------------------------------------------------------------
# Two-physical-leg → paired single-physical-leg splitting, inverse of
# `contract_pairs`.
# ------------------------------------------------------------------------------

"""
    SplitEnvironmentPolicy(; normal=DenseTransferPolicy(), mixed=DenseTransferPolicy())

Choose the transfer representation independently for the split normal
environment `⟨approximation|approximation⟩` and mixed environment
`⟨approximation|target⟩`. Keeping these policies
separate permits, for example, an exact normal equation with a compressed
right-hand-side environment.
"""
Configurations.@option struct SplitEnvironmentPolicy{
        N<:AbstractTransferPolicy,
        M<:AbstractTransferPolicy,
    } <: ExperimentLab.AbstractAlgorithmConfig
    normal::N = DenseTransferPolicy()
    mixed::M = DenseTransferPolicy()
end

"""
    ALSSplit(; sweep, solver, initialization=SVDInitialization(),
             environments=SplitEnvironmentPolicy(), regularization=NoRegularization(),
             stopping=NoStopping(),
             error_measure=:local)

Optimize a paired single-physical-leg approximation of a two-physical-leg MPS.
The initial approximation determines all virtual spaces; single-site updates
preserve them. Supply an initial MPS to `split(target, initial, algorithm)` or
use `initialization` to construct one.

`:local` uses the local quadratic estimate at sweep boundaries for stopping.
`:exact` additionally contracts the true squared distance once per completed
sweep and uses that value for stopping. Exact means untruncated contractions,
subject to floating-point roundoff. Both values are recorded separately.
"""
Configurations.@option struct ALSSplit{
        I<:AbstractInitialization, S<:LatticeIteratorLab.AbstractSweep,
        V<:TensorLinearSolveLab.AbstractSolve, E<:SplitEnvironmentPolicy,
        R<:AbstractRegularization, C<:AbstractStoppingCriterion,
    } <: AbstractSplit
    initialization::I = SVDInitialization()
    sweep::S
    solver::V
    environments::E = SplitEnvironmentPolicy()
    regularization::R = NoRegularization()
    stopping::C = NoStopping()
    error_measure::Symbol = :local
    function ALSSplit{I,S,V,E,R,C}(
            initialization::I, sweep::S, solver::V, environments::E,
            regularization::R, stopping::C, error_measure::Symbol,
        ) where {I,S,V,E,R,C}
        error_measure in (:local, :exact) || throw(ArgumentError(
            "error_measure must be :local or :exact, got $error_measure"))
        return new{I,S,V,E,R,C}(
            initialization, sweep, solver, environments, regularization, stopping, error_measure,
        )
    end
end

ALSSplit(initialization::I, sweep::S, solver::V, environments::E,
         regularization::R, stopping::C, error_measure::Symbol) where {I,S,V,E,R,C} =
    ALSSplit{I,S,V,E,R,C}(
        initialization, sweep, solver, environments, regularization, stopping, error_measure,
    )

# JSON represents symbols as strings; the algorithm owns this field conversion.
Configurations.from_dict(::Type{<:ALSSplit}, ::Type{Symbol}, value::AbstractString) = Symbol(value)
