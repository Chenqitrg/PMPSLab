struct SplitProblemIdentity{S,A}
    target_spaces::S
    algorithm::A
end

_target_spaces(target::PeriodicMPS) =
    [(codomain(tensor), domain(tensor)) for tensor in target]

function _split_problem_identity(
        target::PeriodicMPS,
        algorithm::ALSSplit,
    )
    return SplitProblemIdentity(
        _target_spaces(target),
        deepcopy(algorithm),
    )
end

function _matches_split_problem(
        identity::SplitProblemIdentity,
        target::PeriodicMPS,
        algorithm::ALSSplit,
    )
    return identity.target_spaces == _target_spaces(target) &&
           identity.algorithm == algorithm
end
