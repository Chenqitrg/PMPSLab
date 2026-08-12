"""
    MeasurementHistory([values])

Ordered scientific measurement results committed at algorithm-defined safe
boundaries. The history owns its vector storage; `copy` detaches that storage,
while `deepcopy` also detaches mutable result payloads.
"""
mutable struct MeasurementHistory{
        R<:AbstractMeasurementResult,
    } <: AbstractVector{R}
    values::Vector{R}
    function MeasurementHistory(
            values::AbstractVector{R}
        ) where {R<:AbstractMeasurementResult}
        return new{R}(collect(values))
    end
end

MeasurementHistory() = MeasurementHistory(AbstractMeasurementResult[])
MeasurementHistory{R}() where {R<:AbstractMeasurementResult} =
    MeasurementHistory(R[])

Base.IndexStyle(::Type{<:MeasurementHistory}) = IndexLinear()
Base.size(history::MeasurementHistory) = size(history.values)
Base.getindex(history::MeasurementHistory, index::Int) = history.values[index]

function Base.push!(history::MeasurementHistory{R}, result::R) where {R}
    push!(history.values, result)
    return history
end

Base.copy(history::MeasurementHistory) = MeasurementHistory(history.values)
