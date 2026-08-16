"""
    AbstractAlgorithmConfig

Base type for immutable algorithm configuration values. Configurations compare
and hash by concrete type and field values, including nested configurations.
Concrete configurations with fields should expose a keyword constructor so
call sites identify every setting by name. Positional construction is an
implementation detail used only when validation or normalization is required.
"""
abstract type AbstractAlgorithmConfig <: Algorithm end

_algorithm_config_name(config::AbstractAlgorithmConfig) =
    nameof(typeof(config))

_show_algorithm_config_value(io::IO, value) =
    show(IOContext(io, :compact => true), value)

_show_algorithm_config_value(io::IO, value::TruncationStrategy) =
    print(io, nameof(typeof(value)), "(...)")

function _show_algorithm_config_value(io::IO, ::MatrixAlgebraKit.NoTruncation)
    print(io, "notrunc()")
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByOrder,
    )
    if strategy.by === abs && strategy.rev
        print(io, "truncrank(", strategy.howmany, ")")
    else
        print(io, "TruncationByOrder(...)")
    end
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByError,
    )
    if strategy.p != 2 || (!iszero(strategy.atol) && !iszero(strategy.rtol))
        print(io, "TruncationByError(...)")
    elseif iszero(strategy.atol) && iszero(strategy.rtol)
        print(io, "truncerror()")
    elseif iszero(strategy.atol)
        print(io, "truncerror(rtol=", strategy.rtol, ")")
    else
        print(io, "truncerror(atol=", strategy.atol, ")")
    end
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationIntersection,
    )
    for (index, component) in enumerate(strategy.components)
        index > 1 && print(io, " & ")
        _show_algorithm_config_value(io, component)
    end
    return nothing
end

_show_algorithm_config_value(io::IO, algorithm::MatrixAlgebraKit.Algorithm) =
    print(io, MatrixAlgebraKit.name(algorithm))

_show_algorithm_config_value(io::IO, algorithm::MatrixAlgebraKit.AbstractAlgorithm) =
    print(io, nameof(typeof(algorithm)))

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
            _show_algorithm_config_value(io, value)
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
    if get(io, :compact, false)::Bool
        show(io, config)
        return nothing
    end

    fields = fieldnames(typeof(config))
    if isempty(fields)
        show(io, config)
    else
        println(io, _algorithm_config_name(config), ":")
        _show_algorithm_config_fields(io, config, "")
    end
    return nothing
end

function Base.show(io::IO, config::AbstractAlgorithmConfig)
    print(io, _algorithm_config_name(config))
    print(io, isempty(fieldnames(typeof(config))) ? "()" : "(...)")
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
