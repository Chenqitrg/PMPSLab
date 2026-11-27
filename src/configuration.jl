# Configurations.@option supplies fieldwise equality; keep hashing consistent with it.
function _hash_fields(config::T, seed::UInt)::UInt where {T}
    result = hash(T, seed)
    for field in fieldnames(T)
        result = hash(getfield(config, field), result)
    end
    return result
end

Base.hash(config::AbstractSplit, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::AbstractStoppingCriterion, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::AbstractInitialization, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::AbstractRegularization, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::AbstractTransferPolicy, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::AbstractGaugeFix, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::GeometricDecay, seed::UInt)::UInt = _hash_fields(config, seed)
Base.hash(config::SplitEnvironmentPolicy, seed::UInt)::UInt = _hash_fields(config, seed)
