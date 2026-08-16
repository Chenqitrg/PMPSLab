"""
    @setconfig config path/to/field = value
    @setconfig config begin
        path/to/field = value
        field = value
    end

Return a rebuilt algorithm configuration with the selected fields replaced.
Paths follow the same root-to-leaf order as the configuration tree. Every
affected configuration is rebuilt through its keyword constructor, preserving
validation while allowing concrete field types to change. `config` itself is
not mutated.
"""
macro setconfig(config, changes)
    assignments = _config_assignments(changes)
    result = esc(config)
    for (path, value) in assignments
        result = :(_setconfig($result, $(QuoteNode(path)), $(esc(value))))
    end
    return result
end

function _config_assignments(expression)
    expressions = expression isa Expr && expression.head == :block ?
        filter(item -> !(item isa LineNumberNode), expression.args) :
        (expression,)
    return map(expressions) do assignment
        assignment isa Expr && assignment.head == :(=) || throw(ArgumentError(
            "@setconfig expects `path/to/field = value` assignments"
        ))
        return _config_macro_path(assignment.args[1]), assignment.args[2]
    end
end

_config_macro_path(field::Symbol) = (field,)

function _config_macro_path(expression)
    expression isa Expr && expression.head == :call &&
        length(expression.args) == 3 && expression.args[1] == :/ ||
        throw(ArgumentError(
            "@setconfig paths must contain field names separated by `/`"
        ))
    return (
        _config_macro_path(expression.args[2])...,
        _config_macro_path(expression.args[3])...,
    )
end

function _setconfig(
        config::AbstractAlgorithmConfig,
        path::Tuple{Vararg{Symbol}},
        replacement,
    )
    field = first(path)
    field in fieldnames(typeof(config)) || throw(ArgumentError(
        "$(nameof(typeof(config))) has no configuration field `$field`"
    ))

    if length(path) > 1
        child = getfield(config, field)
        child isa AbstractAlgorithmConfig || throw(ArgumentError(
            "$(nameof(typeof(config))).$field is not a nested algorithm configuration"
        ))
        replacement = _setconfig(child, Base.tail(path), replacement)
    end
    return _replace_config_field(config, field, replacement)
end

function _replace_config_field(
        config::AbstractAlgorithmConfig,
        field::Symbol,
        replacement,
    )
    names = fieldnames(typeof(config))
    values = ntuple(length(names)) do index
        name = names[index]
        name == field ? replacement : getfield(config, name)
    end
    constructor = Base.typename(typeof(config)).wrapper
    return constructor(; NamedTuple{names}(values)...)
end
