"""
    combine_truncation(left, right)

Combine two truncation policies using "either policy may discard" semantics.
MatrixAlgebraKit describes a policy by the singular values it keeps, so this is
the intersection `left & right`, not the union `left | right`.

Compatible rank and error policies are simplified directly to avoid growing a
tree of intersection strategies during repeated transfer multiplication.
"""
function combine_truncation(
        left::TruncationStrategy,
        right::TruncationStrategy,
    )
    left == right && return left
    return left & right
end

function combine_truncation(
        left::MatrixAlgebraKit.TruncationByOrder,
        right::MatrixAlgebraKit.TruncationByOrder,
    )
    left == right && return left
    if left.by == right.by && left.rev == right.rev
        return truncrank(
            min(left.howmany, right.howmany);
            by = left.by,
            rev = left.rev,
        )
    end
    return left & right
end

function combine_truncation(
        left::MatrixAlgebraKit.TruncationByError,
        right::MatrixAlgebraKit.TruncationByError,
    )
    left == right && return left
    if left.p == right.p
        return truncerror(
            atol = max(left.atol, right.atol),
            rtol = max(left.rtol, right.rtol),
            p = left.p,
        )
    end
    return left & right
end
