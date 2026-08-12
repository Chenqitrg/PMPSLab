"""
    AbstractMeasurement <: AbstractAlgorithmConfig

Immutable configuration describing a scientific measurement performed during
an algorithm run.
"""
abstract type AbstractMeasurement <: AbstractAlgorithmConfig end

"""
    AbstractMeasurementResult

Result produced by an [`AbstractMeasurement`](@ref).
"""
abstract type AbstractMeasurementResult end

"""
    measure(measurement::AbstractMeasurement, inputs...)

Perform `measurement` on algorithm-specific `inputs`. Concrete measurement
types define the supported methods.
"""
function measure end
