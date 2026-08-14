"""
    AbstractAlgorithmConfig

Base type for immutable algorithm configuration values. Configurations compare
and hash by concrete type and field values, including nested configurations.
"""
abstract type AbstractAlgorithmConfig <: Algorithm end

_algorithm_config_name(config::AbstractAlgorithmConfig) =
    nameof(typeof(config))

function _show_algorithm_config_fields(
        io::IO,
        config::AbstractAlgorithmConfig,
        prefix::AbstractString,
    )
    fields = fieldnames(typeof(config))
    for (index, field) in enumerate(fields)
        islast = index == length(fields)
        branch = islast ? "└─ " : "├─ "
        child_prefix = prefix * (islast ? "   " : "│  ")
        value = getfield(config, field)

        print(io, prefix, branch, field, ": ")
        if value isa AbstractAlgorithmConfig
            print(io, _algorithm_config_name(value))
            child_fields = fieldnames(typeof(value))
            if isempty(child_fields)
                println(io, "()")
            else
                println(io)
                _show_algorithm_config_fields(io, value, child_prefix)
            end
        else
            show(IOContext(io, :compact => true), value)
            println(io)
        end
    end
    return nothing
end

function Base.show(
        io::IO,
        ::MIME"text/plain",
        config::AbstractAlgorithmConfig,
    )
    name = _algorithm_config_name(config)
    if get(io, :compact, false)::Bool
        print(io, name, "(...)")
        return nothing
    end

    fields = fieldnames(typeof(config))
    if isempty(fields)
        print(io, name, "()")
    else
        println(io, name, ":")
        _show_algorithm_config_fields(io, config, "")
    end
    return nothing
end

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
