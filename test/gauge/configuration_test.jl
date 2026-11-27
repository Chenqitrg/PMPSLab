using Configurations: to_dict, from_dict
using TensorKit: MatrixAlgebraKit

@testset "projector options preserve truncation" begin
    for original in (ProjectorGaugeFix(), ProjectorGaugeFix(tol=1e-13, trunc=truncrank(4))),
        include_defaults in (true, false)
        # Configurations keeps typed options in memory; binary storage is an application choice.
        parameters = to_dict(original; include_defaults)
        @test isequal(from_dict(typeof(original), parameters), original)
    end
end
