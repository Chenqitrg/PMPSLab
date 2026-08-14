@testset "public method ambiguities" begin
    @test isempty(detect_ambiguities(PMPSLab; recursive=true))
end
