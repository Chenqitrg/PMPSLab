module PMPSLab

using TensorKit
using Configurations: @option
using LinearAlgebra: normalize, normalize!
using MPSKit: GenericMPSTensor, PeriodicVector, TruncationStrategy
using ExperimentLab: ExperimentLab, AbstractAlgorithmConfig
import TensorKit: scalartype
import MPSKit: left_virtualspace, right_virtualspace, physicalspace
import MPSKit, Random, LinearAlgebra, PeriodicProductsLab, TransferMatricesLab

# Representation and tensor operations.
export PeriodicMPS, eachsite, physicalspace, physicalspaces
export left_virtualspace, right_virtualspace, assertconsistency, dense
export contract_pairs, map_tensors, mul_left_virtual, mul_right_virtual
export insert_gauge!, standardize, one_mps, random_cdl, random_cdl_mps, lift

# Periodic-MPS numerical operations and their policies.
export ProjectorGaugeFix, gaugefix!
export PeriodicTransferMatrix, DenseTransferPolicy, TruncatedTransferPolicy
export transfer_at, expectation, squared_norm, squared_distance, normalize, normalize!
export build_transfer_cache, rebuild_transfer_cache!
public AbstractGaugeFix, P_decomp, find_projectors, AbstractTransferPolicy
public rebuild_left_transfer_cache!, rebuild_right_transfer_cache!

include("representation/container.jl")
include("representation/show.jl")
include("representation/constructors.jl")
include("representation/cdl.jl")
include("contractions/contraction.jl")
include("representation/lift.jl")

include("gauge/pseudopow.jl")
include("gauge/projector.jl")
include("gauge/interface.jl")

include("environments/periodic.jl")
include("environments/policy.jl")
include("environments/distance.jl")
include("environments/show.jl")
include("configuration.jl")

end
