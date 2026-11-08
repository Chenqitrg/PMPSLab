module PMPSLab

using TensorKit, LinearAlgebra
using Configurations: @option
using MPSKit: GenericMPSTensor, PeriodicVector, TruncationStrategy
using ExperimentLab: ExperimentLab, AbstractAlgorithmConfig
using ResumableFunctions: ResumableFunctions, @resumable, @yield
import Base: split
import TensorKit: scalartype
import MPSKit: left_virtualspace, right_virtualspace, physicalspace
import MPSKit, Random, Configurations, MatrixAlgebraKit
import PeriodicProductsLab, TensorLinearSolveLab, LatticeIteratorLab

# Construct and inspect periodic MPS values.
export PeriodicMPS, eachsite, physicalspace, physicalspaces
export left_virtualspace, right_virtualspace, assertconsistency, dense
export contract_pairs, map_tensors, mul_left_virtual, mul_right_virtual
export insert_gauge!, standardize, one_mps, random_cdl, random_cdl_mps, lift

# Numerical operations and the default optimization entry point.
export ProjectorGaugeFix, gaugefix!
export PeriodicTransferMatrix, DenseTransferPolicy, TruncatedTransferPolicy
export transfer_at, expectation, squared_norm, squared_distance, normalize, normalize!
export build_transfer_cache, rebuild_transfer_cache!
export split, ALSSplit, initial_approximation

# Specialized policies and algorithm-author interfaces use qualified names.
public DenseTransferMatrix, TruncatedTransferMatrix
public AbstractGaugeFix, P_decomp, find_projectors, AbstractTransferPolicy
public rebuild_left_transfer_cache!, rebuild_right_transfer_cache!
public RandomInitialization, IdentityInitialization, SVDInitialization, GaugeInitialization
public SplitEnvironmentPolicy, NuclearNorm, NoRegularization, GeometricDecay
public NoStopping, CostTolerance, CostConvergence
public AbstractSplit, AbstractSplitState, AbstractInitialization
public AbstractStoppingCriterion, check_convergence
public SplitState, initialize_split, SplitProgress
public split_steps, SplitSnapshot, restore_split

include("representation/container.jl")
include("representation/show.jl")
include("representation/constructors.jl")
include("representation/cdl.jl")
include("contractions/contraction.jl")
include("representation/lift.jl")

include("transfers/dense.jl")
include("transfers/truncation.jl")
include("transfers/truncated.jl")
include("transfers/show.jl")

include("gauge/pseudopow.jl")
include("gauge/projector.jl")
include("gauge/interface.jl")
include("environments/periodic.jl")
include("environments/policy.jl")
include("environments/distance.jl")
include("environments/show.jl")

include("optimization/interface.jl")
include("optimization/algorithm/initialization.jl")
include("optimization/algorithm/stopping.jl")
include("optimization/algorithm/regularization.jl")
include("optimization/algorithm/config.jl")
include("optimization/als/workspace.jl")
include("optimization/als/local_equation.jl")
include("optimization/als/regularization.jl")
include("optimization/als/update.jl")
include("optimization/als/iteration.jl")
include("execution/run.jl")
include("execution/snapshot.jl")
include("configuration.jl")

end
