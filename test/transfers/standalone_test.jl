using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix
@testset "integrated transfer ownership and materialization" begin
    loaded = Set(id.name for id in keys(Base.loaded_modules))
    @test isempty(intersect(loaded, Set(["MPSSplitLab", "TransferMatricesLab", "TransferEnvironmentsLab"])))
    @test parentmodule(DenseTransferMatrix) === PMPSLab
    @test parentmodule(TruncatedTransferMatrix) === PMPSLab
    tensor = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
    exact = DenseTransferMatrix(tensor)
    factored = TruncatedTransferMatrix(exact, TruncationStrategy())
    @test TensorMap(exact) === exact.data
    @test convert(TensorMap, exact) === exact.data
    @test convert(TensorMap, factored) ≈ TensorMap(exact)
    @test isempty(detect_ambiguities(PMPSLab))
end
