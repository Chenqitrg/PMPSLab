# ------------------------------------------------------------------------------
# Nuclear norm regularization approach to loop canonicalization.
# TODO: implement
# ------------------------------------------------------------------------------

@kwdef struct NuclearNormCanonical <: AbstractLoopCanonical
    tol::Float64 = 1e-12
    maxiter::Int = 100
    trscheme::TruncationStrategy = TruncationStrategy()
    ρ::Float64 = 0.8
    ξ_init::Float64 = 1e-5
    ξ_min::Float64 = 1e-7
end
