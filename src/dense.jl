"""
    dense(mps::PeriodicMPS)

Contract all site tensors around the loop into a single dense tensor with
all physical legs open. The virtual indices are fully traced out.

**Warning**: O(D^N) cost — only use for small test systems.

For a 2-site MPS with tensors `T[i]` of shape `(L_i ⊗ p_i ← R_i)`, the N-site
loop trace is `∑_{v_1…v_N} T[1]^{p_1}_{v_1,v_2} · T[2]^{p_2}_{v_2,v_3} · … · T[N]^{p_N}_{v_N,v_1}`.
"""
function dense(mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    left = id(E, left_virtualspace(mps[1]))
    for T in mps
        left_tmp = left * transpose(T, ((1,), (3, 2)))
        out_legs = numout(left_tmp)
        in_legs_inds = Tuple(1 : out_legs)
        left = transpose(left_tmp, ((in_legs_inds..., out_legs + 2), (out_legs + 1,)))
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

function dense(mps::PeriodicMPS{<:AbstractTensorMap{E,S,3,1}}) where {E,S}
    left = id(E, left_virtualspace(mps[1]))
    for T in mps
        left_tmp = left * transpose(T, ((1,), (4, 3, 2)))
        out_legs = numout(left_tmp)
        in_legs_inds = Tuple(1 : out_legs)
        left = transpose(left_tmp, ((in_legs_inds..., out_legs + 3, out_legs + 2), (out_legs + 1,)))
    end
    # left = L₁ ⊗ P_{1,b} ⊗ P_{1,a} ⊗ ... ⊗ P_{N,b} ⊗ P_{N,a} ← L₁  → trace out L₁
    N_cod = length(codomain(left))
    phys_codom = ProductSpace(ntuple(i -> codomain(left)[i + 1], N_cod - 1))
    result = zeros(E, phys_codom)
    TensorKit.planartrace!(result, left,
        (Tuple(2:N_cod), ()),   # (p₁, p₂): physical legs → output
        ((1,), (N_cod + 1,)),   # (q₁, q₂): trace L₁(codim) ↔ L₁(dom)
        1.0, 0.0)
    return result
end
