@testset "DenseTransferMatrix — 1-site ket constructor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    E = DenseTransferMatrix(T)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := T[-1 1; -3] * T'[-4; -2 1]
    @test E ≈ manual
end

@testset "DenseTransferMatrix — MPSKit transfer interface" begin
    D, d = 3, 2
    tensor = randn(ComplexF64, ℂ^D ⊗ ℂ^d, ℂ^D)
    transfer = DenseTransferMatrix(tensor)
    bond_tensor = randn(ComplexF64, ℂ^D, ℂ^D)

    @test transfer isa AbstractTransferMatrix
    @test dense(transfer) === transfer.data
    @test transfer * bond_tensor ≈
          TransferMatrix(tensor, tensor) * bond_tensor
end

@testset "DenseTransferMatrix — 2-site ket constructor" begin
    D, d = 3, 2
    T = randn(ComplexF64, ℂ^D ⊗ ℂ^d ⊗ ℂ^d, ℂ^D)
    E = DenseTransferMatrix(T)

    @test length(codomain(E)) == 2
    @test length(domain(E)) == 2
    @test codomain(E) == ℂ^D ⊗ (ℂ^D)'

    @plansor manual[-1 -2; -3 -4] := T[-1 1 2; -3] * T'[-4; -2 1 2]
    @test E ≈ manual
end

@testset "transfer constructors — mixed 2-site/1-site tensors" begin
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
    @test dense(TruncatedTransferMatrix(A, B1, B2, TruncationStrategy())) ≈
          dense(E)
end
