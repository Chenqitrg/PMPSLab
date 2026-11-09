# Adapted from TNRKit.jl's src/schemes/btrg.jl (pseudopow).
# Copyright (c) 2025 Victor Vanthilt. MIT license; see THIRD_PARTY_NOTICES.md.
# Pseudo-power: suppresses small singular values to avoid 1/0 blowup
function pseudopow(t::DiagonalTensorMap, a::Real; tol = eps(scalartype(t))^(3 / 4))
    t′ = copy(t)
    for (_, b) in blocks(t′)
        @inbounds for i in 1:size(b, 1)
            b.diag[i] = abs(b.diag[i]) < tol ? b.diag[i] : b.diag[i]^a
        end
    end
    return t′
end
