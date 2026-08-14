"""
    AbstractFixedPointMeasurement <: AbstractMeasurement

Measurement evaluated from a normalized tensor cell at an RG boundary.
"""
abstract type AbstractFixedPointMeasurement <: AbstractMeasurement end

"""
    NoFixedPointMeasurement()

Disable the optional fixed-point measurement at LoopTNR RG boundaries.
"""
struct NoFixedPointMeasurement <: AbstractFixedPointMeasurement end

"""
    TNRMeasurement(;
        normalization=InfinityNormNormalization(),
        fixed_point=NoFixedPointMeasurement(),
    )

Configuration of the scientific measurements performed at each LoopTNR RG
boundary.
"""
@kwdef struct TNRMeasurement{
        N<:AbstractNormalization,
        F<:AbstractFixedPointMeasurement,
    } <: AbstractMeasurement
    normalization::N = InfinityNormNormalization()
    fixed_point::F = NoFixedPointMeasurement()
end

"""
    TNRMeasurementResult(rg, normalization, fixed_point)

Scientific measurement result committed at one LoopTNR RG boundary.
"""
struct TNRMeasurementResult{
        N<:NormalizationResult,
        F<:Union{Nothing,AbstractMeasurementResult},
    } <: AbstractMeasurementResult
    rg::Int
    normalization::N
    fixed_point::F
end

function _fixed_point_measurement(
        ::NoFixedPointMeasurement,
        ::AbstractTensorCell,
    )
    return nothing
end

function _fixed_point_measurement(
        measurement::AbstractFixedPointMeasurement,
        cell::AbstractTensorCell,
    )
    result = measure(measurement, cell)
    result isa AbstractMeasurementResult || throw(ArgumentError(
        "a fixed-point measurement must return an AbstractMeasurementResult, " *
        "got $(typeof(result))"
    ))
    return result
end
