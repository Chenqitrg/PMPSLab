"""
    dense(mps::PeriodicMPS)

Contract all site tensors around the loop into a single dense tensor with
all physical legs open. The virtual indices are fully traced out.

**Warning**: the output space grows exponentially with the number of physical
legs, so this is intended only for small test systems.

For a 2-site MPS with tensors `T[i]` of shape `(L_i ⊗ p_i ← R_i)`, the N-site
loop trace is `∑_{v_1…v_N} T[1]^{p_1}_{v_1,v_2} · T[2]^{p_2}_{v_2,v_3} · … · T[N]^{p_N}_{v_N,v_1}`.
"""
function dense(mps::PeriodicMPS{<:AbstractTensorMap{E,S,M,1}}) where {E,S,M}
    left = id(E, left_virtualspace(mps[1]))
    for T in mps
        # Move the right virtual leg and the physical legs to the domain.  The
        # physical legs are reversed here and restored below, so their public
        # order remains site-major for any number of legs per tensor.
        left_tmp = left * transpose(T, ((1,), (M + 1, reverse(2:M)...)))
        out_legs = numout(left_tmp)
        old_outputs = Tuple(1:out_legs)
        new_physical = Tuple(reverse((out_legs + 2):(out_legs + M)))
        left = transpose(
            left_tmp, ((old_outputs..., new_physical...), (out_legs + 1,))
        )
    end
    # left = L₁ ⊗ P₁ ⊗ ... ⊗ P_N ← L₁  → trace out L₁
    N_cod = length(codomain(left))
    phys_codom = ProductSpace(ntuple(i -> codomain(left)[i + 1], N_cod - 1))
    result = zeros(E, phys_codom)
    TensorKit.planartrace!(result, left,
        (Tuple(2:N_cod), ()),   # (p₁, p₂): physical legs → output
        ((1,), (N_cod + 1,)),   # (q₁, q₂): trace L₁(codim) ↔ L₁(dom)
        1.0, 0.0)
    return result
end

"""
    pair_contraction(mps::PeriodicMPS) -> PeriodicMPS

Coarse-grain `mps` by contracting adjacent pairs of sites: (1,2), (3,4), ….
Each pair (T_{2k-1}, T_{2k}) becomes one two-site tensor

    T_k' : L_{2k-1} ⊗ P_{2k-1} ⊗ P_{2k} ← R_{2k}

summing over the shared virtual leg (R_{2k-1} = L_{2k}). The result is a
two-site MPS with N/2 sites, each carrying two physical legs in loop order.
Requires N even.

The full loop product is preserved: `dense(pair_contraction(mps))` equals
`dense(mps)` with the same physical leg order.
"""
function pair_contraction(mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    N = length(mps)
    iseven(N) || throw(ArgumentError(
        "pair contraction requires an even number of sites, got $N"
    ))
    paired = [_pair_contract(mps[2k - 1], mps[2k]) for k in 1:(N ÷ 2)]
    return PeriodicMPS(paired)
end

# Contract two adjacent one-site tensors into one two-site tensor,
# summing over the shared virtual leg.
function _pair_contract(T1::TensorMap{E, S, 2, 1}, T2::TensorMap{E, S, 2, 1}) where {E, S}
    @plansor T12[-1 -2 -3; -4] := T1[-1 -2; 1] * T2[1 -3; -4]
    return T12
end
