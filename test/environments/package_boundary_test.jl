using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix
@testset "transfer layer ownership and materialization" begin
    @test parentmodule(DenseTransferMatrix) === PMPSLab
    @test parentmodule(TruncatedTransferMatrix) === PMPSLab
    @test parentmodule(PeriodicTransferMatrix) === PMPSLab
    @test Base.ispublic(PMPSLab, :DenseTransferMatrix)
    @test Base.ispublic(PMPSLab, :TruncatedTransferMatrix)
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    transfer = transfer_at(mps, 1)
    factored = transfer_at(mps, 1, TruncationStrategy())
    @test dense(transfer) === TensorMap(transfer)
    @test dense(factored) ≈ TensorMap(factored)
    @test isempty(detect_ambiguities(PMPSLab, PeriodicProductsLab))
end
