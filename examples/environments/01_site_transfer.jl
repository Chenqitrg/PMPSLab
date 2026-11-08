# Run from PMPSLab: julia --project=. examples/environments/01_site_transfer.jl
using PMPSLab: DenseTransferMatrix
# Construct a transfer matrix from one MPS site.
using PMPSLab: PeriodicMPS
using PMPSLab: transfer_at
using Test: @test

# Arguments: scalar type, number of sites, physical dimension, virtual dimension.
mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
site = 2

# Choose the site tensor, then let PMPSLab perform the contraction.
E = transfer_at(mps, site)
@test E ≈ DenseTransferMatrix(mps[site])
@test E ≈ transfer_at(mps, mps, site)
display(E)

# For two MPS, the order is bra, ket, site; the bra tensor is conjugated.
ket = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
mixed = transfer_at(mps, ket, site)
@test mixed ≈ DenseTransferMatrix(mps[site], ket[site])
nothing
