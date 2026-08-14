"""
    AbstractAlgorithmConfig

Base type for immutable algorithm configuration values. Configurations compare
and hash by concrete type and field values, including nested configurations.
"""
abstract type AbstractAlgorithmConfig <: Algorithm end

function Base.:(==)(left::T, right::T) where {T<:AbstractAlgorithmConfig}
    return all(
        field -> getfield(left, field) == getfield(right, field),
        fieldnames(T),
    )
end

function Base.isequal(left::T, right::T) where {T<:AbstractAlgorithmConfig}
    return all(
        field -> isequal(getfield(left, field), getfield(right, field)),
        fieldnames(T),
    )
end

function Base.hash(config::T, seed::UInt) where {T<:AbstractAlgorithmConfig}
    result = hash(T, seed)
    for field in fieldnames(T)
        result = hash(getfield(config, field), result)
    end
    return result
end
