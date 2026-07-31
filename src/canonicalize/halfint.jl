# ------------------------------------------------------------------------------
# Half-integer index for bond/link positions
# ------------------------------------------------------------------------------

struct HalfInt
    n::Int  # represents n + 1/2, i.e. the bond between site n and site n+1
end
Base.convert(::Type{HalfInt}, n::Int) = HalfInt(n)
Base.hash(h::HalfInt, h0::UInt) = hash(h.n, h0)
Base.:(==)(a::HalfInt, b::HalfInt) = a.n == b.n
Base.:+(h::HalfInt, n::Int) = HalfInt(h.n + n)
Base.:-(h::HalfInt, n::Int) = HalfInt(h.n - n)
