# Run from PMPSLab: julia --project=. examples/environments/02_periodic_product.jl
# Collect the site transfers for one period and multiply them in order.
using PMPSLab: PeriodicMPS
using PMPSLab: PeriodicTransferMatrix, transfer_at
using Test: @test

mps = rand(PeriodicMPS, ComplexF64, 4, 2, 2)
transfers = PeriodicTransferMatrix(mps)

# transfers is a periodic container, not a prefix/suffix cache.
# Its entries are computed at construction; later MPS changes do not update them.
@test length(transfers) == 4
@test transfers[2] ≈ transfer_at(mps, 2)
@test transfers[5] === transfers[1]
display(transfers)

# The single-argument *(transfers) computes the ordered full-period product.
product = *(transfers)
@test product ≈ transfers[1] * transfers[2] * transfers[3] * transfers[4]
display(product)
nothing
