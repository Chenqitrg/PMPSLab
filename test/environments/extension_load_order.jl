# Run separately in the package environment; each case has a fresh Julia process.
using Test

@testset "optional dense extension in both load orders" begin
    project = dirname(@__DIR__)
    for order in ("using PMPSLab; using TransferMatricesLab", "using TransferMatricesLab; using PMPSLab")
        code = order * """
        ; using TensorKit
        A = randn(ComplexF64, ℂ^2 ⊗ ℂ^2, ℂ^2)
        E = TransferMatricesLab.DenseTransferMatrix(A)
        @assert Base.get_extension(TransferMatricesLab, :TransferMatricesPMPSLabExt) !== nothing
        @assert PMPSLab.dense(E) === TensorMap(E)
        F = TransferMatricesLab.TruncatedTransferMatrix(E, truncrank(2))
        @assert PMPSLab.dense(F) ≈ TensorMap(F)
        """
        @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`)
    end
end
nothing
