@testset "cross-package method ambiguities" begin
    @test isempty(detect_ambiguities(PMPSLab, PeriodicProductsLab, ExperimentLab;
        recursive=true))
end
