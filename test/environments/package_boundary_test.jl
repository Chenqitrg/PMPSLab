@testset "transfer layer ownership and optional dense bridge" begin
    @test parentmodule(DenseTransferMatrix) === TransferMatricesLab
    @test parentmodule(TruncatedTransferMatrix) === TransferMatricesLab
    @test parentmodule(PeriodicTransferMatrix) === PMPSLab
    @test !(:DenseTransferMatrix in names(PMPSLab))
    @test !(:TruncatedTransferMatrix in names(PMPSLab))
    @test Base.get_extension(TransferMatricesLab, :TransferMatricesPMPSLabExt) !== nothing
    mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
    transfer = transfer_at(mps, 1)
    factored = transfer_at(mps, 1, TruncationStrategy())
    @test dense(transfer) === TensorMap(transfer)
    @test dense(factored) ≈ TensorMap(factored)
    @test isempty(detect_ambiguities(TransferMatricesLab, PMPSLab, PeriodicProductsLab))
end
