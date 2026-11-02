# Corner-double-line tensors and periodic CDL MPS test states.

"""
    random_cdl(Vdl, Vdr, Vlu, Vld, Vru, Vrd)
    random_cdl(elt, Vdl, Vdr, Vlu, Vld, Vru, Vrd)

Construct a local corner-double-line tensor from three independent random maps

```
Vdl ← Vld,    Vrd ← Vdr,    Vlu ← Vru.
```

The open legs are bent into the layout

```
Vlu ⊗ Vld' ⊗ Vdl ⊗ Vdr' ← Vru ⊗ Vrd'.
```
"""
random_cdl(
    Vdl::ElementarySpace, Vdr::ElementarySpace,
    Vlu::ElementarySpace, Vld::ElementarySpace,
    Vru::ElementarySpace, Vrd::ElementarySpace,
) = random_cdl(ComplexF64, Vdl, Vdr, Vlu, Vld, Vru, Vrd)

function random_cdl(
        elt::Type{<:Number},
        Vdl::ElementarySpace, Vdr::ElementarySpace,
        Vlu::ElementarySpace, Vld::ElementarySpace,
        Vru::ElementarySpace, Vrd::ElementarySpace,
    )
    down = rand(elt, Vdl ← Vld)
    lower_right = rand(elt, Vrd ← Vdr)
    upper = rand(elt, Vlu ← Vru)
    @plansor tensor[-1 -2 -3 -4; -5 -6] :=
        upper[-1; -5] * down[-3; -2] * lower_right[-6; -4]
    return tensor
end

"""
    random_cdl_mps(elt, N, χ) -> PeriodicMPS
    random_cdl_mps(elt, N, V) -> PeriodicMPS

Construct an `N`-site periodic CDL MPS. Each local tensor is built with
`random_cdl`; the two virtual double lines are fused into a bond space of
dimension `dim(V)^2`. TensorKit unitary fusion maps are inserted on the two
sides of every bond as mutually inverse pairs, so they do not change the
contracted periodic state.
"""
random_cdl_mps(elt::Type{<:Number}, N::Int, χ::Int) =
    random_cdl_mps(elt, N, ℂ^χ)

function random_cdl_mps(
        elt::Type{<:Number}, N::Int, V::ElementarySpace,
    )
    N > 0 || throw(ArgumentError("N must be positive"))

    virtual = fuse(V ⊗ V')
    fusion = unitary(elt, V ⊗ V', virtual)
    tensors = map(1:N) do _
        tensor = random_cdl(elt, V, V, V, V, V, V)
        @plansor mps_tensor[-1 -2 -3; -4] :=
            fusion'[-1; 1 2] * tensor[1 2 -2 -3; 3 4] * fusion[3 4; -4]
        return mps_tensor
    end
    return PeriodicMPS(tensors)
end
