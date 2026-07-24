"""
    dense(mps::PeriodicMPS)

Contract all site tensors around the loop into a single dense tensor with
all physical legs open. The virtual indices are fully traced out.

**Warning**: O(D^N) cost — only use for small test systems.

For a 2-site MPS with tensors `T[i]` of shape `(L_i ⊗ p_i ← R_i)`, the N-site
loop trace is `∑_{v_1…v_N} T[1]^{p_1}_{v_1,v_2} · T[2]^{p_2}_{v_2,v_3} · … · T[N]^{p_N}_{v_N,v_1}`.
"""
function dense(mps::PeriodicMPS)
    T = mps[1]
    return result
end
