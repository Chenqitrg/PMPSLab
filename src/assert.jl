"""
    assertconsistency(mps::PeriodicMPS)

Check that the virtual spaces of adjacent site tensors are compatible around
the loop. Throws `TensorKit.SpaceMismatch` on failure.
"""
function assertconsistency(mps::PeriodicMPS)
    tensors = mps.tensors
    N = length(tensors)
    for i in 1:N
        L = left_virtualspace(tensors[i])
        R = right_virtualspace(tensors[i - 1])
        L == R || throw(TensorKit.SpaceMismatch(
            "left_virtualspace at site $i ($L) ≠ right_virtualspace at site $(i-1) ($R)"
        ))
    end
    return nothing
end
