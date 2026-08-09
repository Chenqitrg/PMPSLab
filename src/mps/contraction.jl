# Full-period contraction and adjacent-site coarse graining.

"""
    dense(mps::PeriodicMPS)

Contract one full period into a single dense tensor with all physical legs
open. The virtual indices are fully traced out.

**Warning**: the output space grows exponentially with the number of physical
legs, so this is intended only for small test systems.

For a 2-site MPS with tensors `T[i]` of shape `(L_i ⊗ p_i ← R_i)`, the
N-site periodic trace is `∑_{v_1…v_N} T[1]^{p_1}_{v_1,v_2} ·
T[2]^{p_2}_{v_2,v_3} · … · T[N]^{p_N}_{v_N,v_1}`.
"""
function dense(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,M,1}},
    ) where {E,S,M}
    left = id(E, left_virtualspace(first(mps)))
    for tensor in mps
        # Move the right virtual leg and the physical legs to the domain. The
        # physical legs are reversed here and restored below, so their public
        # order remains site-major for any number of legs per tensor.
        left_tmp = left * repartition(tensor, 1, M)
        left = repartition(left_tmp, numind(left_tmp) - 1, 1)
    end

    # left = L₁ ⊗ P₁ ⊗ ... ⊗ P_N ← L₁; trace out L₁.
    codomain_count = length(codomain(left))
    physical_codomain = ProductSpace(ntuple(
        i -> codomain(left)[i + 1], codomain_count - 1
    ))
    result = zeros(E, physical_codomain)
    TensorKit.planartrace!(
        result,
        left,
        (Tuple(2:codomain_count), ()),
        ((1,), (codomain_count + 1,)),
        1.0,
        0.0,
    )
    return result
end

"""
    pair_contraction(mps::PeriodicMPS) -> PeriodicMPS

Coarse-grain `mps` by contracting adjacent pairs of sites: (1,2), (3,4), ….
Each pair `(T_{2k-1}, T_{2k})` becomes one two-site tensor

    T_k' : L_{2k-1} ⊗ P_{2k-1} ⊗ P_{2k} ← R_{2k}

summing over the shared virtual leg. The result contains `N/2` tensors and
requires even `N`. The full-period product and physical-leg order are
preserved.
"""
function pair_contraction(
        mps::PeriodicMPS{<:AbstractTensorMap{E,S,2,1}},
    ) where {E,S}
    N = length(mps)
    iseven(N) || throw(ArgumentError(
        "pair contraction requires an even number of sites, got $N"
    ))
    paired = [_pair_contract(mps[2k - 1], mps[2k]) for k in 1:(N ÷ 2)]
    return PeriodicMPS(paired)
end

function _pair_contract(
        left::TensorMap{E,S,2,1},
        right::TensorMap{E,S,2,1},
    ) where {E,S}
    @plansor paired[-1 -2 -3; -4] :=
        left[-1 -2; 1] * right[1 -3; -4]
    return paired
end
