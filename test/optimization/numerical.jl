using LatticeIteratorLab
using LatticeIteratorLab: SweepState, Forward, Backward
using PeriodicProductsLab
using TensorLinearSolveLab
# Run independently: julia --project=. -e 'using Pkg; Pkg.test(test_args=["numerical"])'
using Test, PMPSLab, ExperimentLab
include("module_boundary_test.jl")
include("accessors_test.jl")
include("split/runtests.jl")
include("configuration_test.jl")

@testset "retired packages are not loaded" begin
    loaded = Set(pkg.name for pkg in keys(Base.loaded_modules))
    @test !("MPSSplitLab" in loaded)
    @test !("TransferMatricesLab" in loaded)
    @test !("GaugeFixingLab" in loaded)
    @test !("TransferEnvironmentsLab" in loaded)
    @test !("ExecutionLab" in loaded)
    @test !("AlgorithmDataLab" in loaded)
end
