# Stable JSON identities live outside numerical configuration structs.
const CONFIG_REGISTRY = ExperimentLab.ConfigRegistry(
    ExperimentLab.ConfigRegistry(
        "PMPSLab.ProjectorGaugeFix" => ProjectorGaugeFix,
        "PMPSLab.DenseTransferPolicy" => DenseTransferPolicy,
        "PMPSLab.TruncatedTransferPolicy" => TruncatedTransferPolicy,
        "PMPSLab.RandomInitialization" => RandomInitialization,
        "PMPSLab.IdentityInitialization" => IdentityInitialization,
        "PMPSLab.SVDInitialization" => SVDInitialization,
        "PMPSLab.GaugeInitialization" => GaugeInitialization,
        "PMPSLab.NoStopping" => NoStopping,
        "PMPSLab.CostTolerance" => CostTolerance,
        "PMPSLab.CostConvergence" => CostConvergence,
        "PMPSLab.NoRegularization" => NoRegularization,
        "PMPSLab.GeometricDecay" => GeometricDecay,
        "PMPSLab.NuclearNorm" => NuclearNorm,
        "PMPSLab.SplitEnvironmentPolicy" => SplitEnvironmentPolicy,
        "PMPSLab.ALSSplit" => ALSSplit,
    ),
    ExperimentLab.config_registry(LatticeIteratorLab.AbstractUpdate),
    ExperimentLab.config_registry(TensorLinearSolveLab.AbstractSolve),
)

ExperimentLab.config_registry(::Type{<:AbstractInitialization}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:AbstractStoppingCriterion}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:AbstractRegularization}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:GeometricDecay}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:SplitEnvironmentPolicy}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:AbstractSplit}) = CONFIG_REGISTRY

ExperimentLab.config_registry(::Type{<:AbstractGaugeFix}) = CONFIG_REGISTRY
ExperimentLab.config_registry(::Type{<:AbstractTransferPolicy}) = CONFIG_REGISTRY
