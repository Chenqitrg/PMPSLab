# JSON identities of the periodic-MPS numerical policies.
const CONFIG_REGISTRY = ExperimentLab.ConfigRegistry(
    "PMPSLab.ProjectorGaugeFix" => ProjectorGaugeFix,
    "PMPSLab.DenseTransferPolicy" => DenseTransferPolicy,
    "PMPSLab.TruncatedTransferPolicy" => TruncatedTransferPolicy,
)
ExperimentLab.config_registry(::Type{<:AbstractGaugeFix}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:AbstractTransferPolicy}) = CONFIG_REGISTRY
