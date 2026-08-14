"""
    AbstractFixedPointMeasurement <: AbstractMeasurement

Measurement evaluated from a normalized tensor cell at an RG boundary.
"""
abstract type AbstractFixedPointMeasurement <: AbstractMeasurement end

"""
    TNRMeasurement([normalization], [fixed_point])

Configuration of the scientific measurements performed at each LoopTNR RG
boundary. `fixed_point=nothing` disables the optional fixed-point measurement.
"""
struct TNRMeasurement{
        N<:AbstractNormalization,
        F<:Union{Nothing,AbstractFixedPointMeasurement},
    } <: AbstractMeasurement
    normalization::N
    fixed_point::F
end

TNRMeasurement() = TNRMeasurement(InfinityNormNormalization(), nothing)

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
        ::Nothing,
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
