using PMPSLab: DenseTransferMatrix, TruncatedTransferMatrix
@testset "DenseTransferMatrix — single-physical-leg ket" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    E = DenseTransferMatrix(T)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := T[-1 1; -3] * T'[-4; -2 1]
    @test E ≈ manual
end

@testset "DenseTransferMatrix — display" begin
    tensor = randn(ComplexF64, ℂ^2 ⊗ ℂ^3, ℂ^2)
    transfer = DenseTransferMatrix(tensor)

    @test summary(transfer) ==
          "4×4 DenseTransferMatrix(ComplexF64, ComplexSpace)"
    @test sprint(show, transfer) == summary(transfer)

    shown = sprint(
        show,
        MIME"text/plain"(),
        transfer;
        context=(:compact => false,),
    )
    @test startswith(shown, summary(transfer) * ":\n")
    @test occursin("ℂ^2 │       │ (ℂ^2)'", shown)
    @test occursin("├───────┤", shown)

    directed = DenseTransferMatrix(randn(
        ComplexF64,
        ℂ^2 ⊗ ℂ^3,
        ℂ^4 ⊗ ℂ^5,
    ))
    directed_shown = sprint(show, MIME"text/plain"(), directed)
    @test occursin("ℂ^4 │       │ ℂ^5", directed_shown)
    @test occursin("ℂ^2 │       │ ℂ^3", directed_shown)
    @test first(findfirst("ℂ^4", directed_shown)) <
          first(findfirst("ℂ^2", directed_shown))

    compact = sprint(
        show,
        MIME"text/plain"(),
        transfer;
        context=(:compact => true,),
    )
    @test compact == summary(transfer)
end

@testset "DenseTransferMatrix — MPSKit transfer interface" begin
    D, d = 3, 2
    tensor = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    transfer = DenseTransferMatrix(tensor)
    bond_tensor = randn(ComplexF64, ℂ^D, ℂ^D)

    @test transfer isa AbstractTransferMatrix
    @test TensorMap(transfer) === transfer.data
    @test transfer * bond_tensor ≈
          TransferMatrix(tensor, tensor) * bond_tensor
end

@testset "DenseTransferMatrix — two-physical-leg ket" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    E = DenseTransferMatrix(T)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := T[-1 1 2; -3] * T'[-4; -2 1 2]
    @test E ≈ manual
end

@testset "transfer constructors — mixed target/approximation tensors" begin
    D, d = 3, 2
    A = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    B1 = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    B2 = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    E = DenseTransferMatrix(A, B1, B2)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := B1[-1 1; 2] * B2[2 3; -3] * A'[-4; -2 1 3]
    @test E ≈ manual
    @test TensorMap(TruncatedTransferMatrix(A, B1, B2, TruncationStrategy())) ≈
          TensorMap(E)
end
