"""
    assertconsistency(mps::PeriodicMPS)

Check that the virtual spaces of adjacent site tensors are compatible around
the loop. Throws `TensorKit.SpaceMismatch` on failure.
"""
function assertconsistency(mps::PeriodicMPS)
    N = length(mps)
    for i in 1:N
        L = left_virtualspace(mps[i])
        R = right_virtualspace(mps[i - 1])
        L == R || throw(TensorKit.SpaceMismatch(
            "left_virtualspace at site $i ($L) ≠ right_virtualspace at site $(i-1) ($R)"
        ))
    end
    return nothing
end
