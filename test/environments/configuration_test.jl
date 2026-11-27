using Configurations: to_dict, from_dict

@testset "transfer options preserve truncation" begin
    for original in (DenseTransferPolicy(), TruncatedTransferPolicy(), TruncatedTransferPolicy(trunc=truncrank(3))),
        include_defaults in (true, false)
        parameters = to_dict(original; include_defaults)
        @test isequal(from_dict(typeof(original), parameters), original)
    end
end
