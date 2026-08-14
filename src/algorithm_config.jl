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

function _show_keyword_arguments(io::IO, arguments)
    print(io, "(")
    for (index, (name, value)) in enumerate(arguments)
        index > 1 && print(io, ", ")
        print(io, name, "=")
        show(IOContext(io, :compact => true), value)
    end
    print(io, ")")
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.NoTruncation,
    )
    print(io, "notrunc()")
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByOrder,
    )
    print(io, "truncrank(", strategy.howmany)
    strategy.by === abs || begin
        print(io, "; by=")
        show(IOContext(io, :compact => true), strategy.by)
    end
    strategy.rev || print(io, strategy.by === abs ? "; rev=false" : ", rev=false")
    print(io, ")")
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByError,
    )
    arguments = Pair{Symbol,Any}[]
    iszero(strategy.atol) || push!(arguments, :atol => strategy.atol)
    iszero(strategy.rtol) || push!(arguments, :rtol => strategy.rtol)
    strategy.p == 2 || push!(arguments, :p => strategy.p)
    print(io, "truncerror")
    _show_keyword_arguments(io, arguments)
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByValue,
    )
    arguments = Pair{Symbol,Any}[]
    iszero(strategy.atol) || push!(arguments, :atol => strategy.atol)
    iszero(strategy.rtol) || push!(arguments, :rtol => strategy.rtol)
    strategy.p == 2 || push!(arguments, :p => strategy.p)
    strategy.by === abs || push!(arguments, :by => strategy.by)
    strategy.keep_below && push!(arguments, :keep_below => true)
    print(io, "trunctol")
    _show_keyword_arguments(io, arguments)
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationByFilter,
    )
    print(io, "truncfilter(")
    show(IOContext(io, :compact => true), strategy.filter)
    print(io, ")")
    return nothing
end

function _show_composite_truncation(io::IO, components, operator::AbstractString)
    for (index, component) in enumerate(components)
        index > 1 && print(io, " ", operator, " ")
        nested = component isa Union{
            MatrixAlgebraKit.TruncationIntersection,
            MatrixAlgebraKit.TruncationUnion,
        }
        nested && print(io, "(")
        _show_algorithm_config_value(io, component)
        nested && print(io, ")")
    end
    return nothing
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationIntersection,
    )
    return _show_composite_truncation(io, strategy.components, "&")
end

function _show_algorithm_config_value(
        io::IO,
        strategy::MatrixAlgebraKit.TruncationUnion,
    )
    return _show_composite_truncation(io, strategy.components, "|")
end

function _show_algorithm_config_value(
        io::IO,
        algorithm::MatrixAlgebraKit.AbstractAlgorithm,
    )
    algorithm_type = typeof(algorithm)
    name = algorithm_type <: MatrixAlgebraKit.Algorithm ?
        algorithm_type.parameters[1] : nameof(algorithm_type)
    print(io, name)
    return nothing
end

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
