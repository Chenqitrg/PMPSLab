using Test, PMPSLab, TensorKit
using LinearAlgebra: mul!, norm
using MPSKit: AbstractTransferMatrix, TransferMatrix, TruncationStrategy
using PMPSLab: combine_truncation
import MatrixAlgebraKit

include("dense_test.jl")
include("truncated_test.jl")
include("standalone_test.jl")
