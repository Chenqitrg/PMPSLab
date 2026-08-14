struct SplitProblemIdentity{S,A}
    target_spaces::S
    algorithm::A
end

_target_spaces(mpsA::PeriodicMPS) =
    [(codomain(tensor), domain(tensor)) for tensor in mpsA]

function _split_problem_identity(
        mpsA::PeriodicMPS,
        algorithm::PeriodicSplit,
    )
    return SplitProblemIdentity(
        _target_spaces(mpsA),
        deepcopy(algorithm),
    )
end

function _matches_split_problem(
        identity::SplitProblemIdentity,
        mpsA::PeriodicMPS,
        algorithm::PeriodicSplit,
    )
    return identity.target_spaces == _target_spaces(mpsA) &&
           identity.algorithm == algorithm
end
