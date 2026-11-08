abstract type AbstractRegularization <: ExperimentLab.AbstractAlgorithmConfig end

"""Ordinary ALS without regularization or automatic cost normalization."""
Configurations.@option struct NoRegularization <: AbstractRegularization end

"""
    GeometricDecay(; initial=1e-5, factor=0.8, minimum=1e-7)

Penalty on sweep k (starting at zero): max(minimum, initial * factor^k).
A complete LatticeIteratorLab sweep is one ring for CircularSweep and one round trip
for ZigzagSweep. Set minimum=0 for the paper's unclamped decay.
"""
Configurations.@option struct GeometricDecay <: ExperimentLab.AbstractAlgorithmConfig
    initial::Float64 = 1e-5
    factor::Float64 = 0.8
    minimum::Float64 = 1e-7
    function GeometricDecay(initial::Real, factor::Real, minimum::Real)
        values = Float64.((initial, factor, minimum))
        all(isfinite, values) || throw(ArgumentError("decay parameters must be finite"))
        xi, rho, floor = values
        xi > 0 || throw(ArgumentError("initial penalty must be positive"))
        0 < rho <= 1 || throw(ArgumentError("decay factor must lie in (0, 1]"))
        0 <= floor <= xi || throw(ArgumentError("minimum must lie in [0, initial]"))
        return new(values...)
    end
end

"""
    NuclearNorm(; schedule=GeometricDecay())

Coupled nuclear-norm regularization: the ADMM penalty and singular-value
threshold are xi, corresponding to nuclear-norm weight mu=xi^2.
Penalize the inserted pair bond (odd sites' right leg, even sites' left leg).
Optimization uses a unit-norm target; tensors remain in the original units,
while reported costs and stopping tolerances refer to normalized squared error.
See Homma et al., https://arxiv.org/abs/2306.17479, Section II.2.
"""
Configurations.@option struct NuclearNorm <: AbstractRegularization
    schedule::GeometricDecay = GeometricDecay()
end

_penalty(::NoRegularization, completed_sweeps::Int) = nothing
function _penalty(regularization::NuclearNorm, completed_sweeps::Int)
    schedule = regularization.schedule
    xi = max(schedule.minimum, schedule.initial * schedule.factor^completed_sweeps)
    xi > 0 || throw(ArgumentError("penalty underflow; use a positive schedule minimum"))
    return xi
end
