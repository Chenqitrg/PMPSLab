@testset "cross-package method ambiguities" begin
    @test isempty(detect_ambiguities(PMPSLab, PeriodicProductsLab, AI;
        recursive=true))
end
