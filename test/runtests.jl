using Test, PMPSLab, Random, TensorKit
using MPSKit: PeriodicVector, left_virtualspace, right_virtualspace
include("site_space_test.jl")
include("mps_test.jl")
include("mps_gauge_test.jl")
include("consistency_test.jl")
include("dense_single_physical_leg_test.jl")
include("contract_pairs_test.jl")
include("dense_two_physical_legs_test.jl")

@testset "datatype ownership and legacy datatype helpers" begin
    @test parentmodule(PeriodicMPS) === PMPSLab
    @test parentmodule(ALSSplit) === PMPSLab
    @test !isdefined(PMPSLab, :start_run)
    @test !isdefined(PMPSLab, :build_cache)
    @test dense(PMPSLab.one_mps2(ComplexF64, 2, 2, 1)) ==
          dense(one_mps(ComplexF64, 2, (2, 2), 1))
    state = one_mps(ComplexF64, 4, 2, 1)
    @test dense(PMPSLab.pair_contraction(state)) == dense(contract_pairs(state))
end

include("gauge/runtests.jl")
include("environments/runtests.jl")

module TransferTests
include("transfers/runtests.jl")
end

module OptimizationTests
include("optimization/runtests.jl")
end
