@testset "split owns numerical state, ExperimentLab owns sessions" begin
    @test parentmodule(PeriodicMPS) === PMPSLab
    @test parentmodule(PMPSLab.SplitSnapshot) === PMPSLab
    @test parentmodule(ExperimentLab.Session) === ExperimentLab
    @test PMPSLab.split === Base.split
    @test !isdefined(PMPSLab, :SplitRun)
    @test !isdefined(PMPSLab, :start_run)
end
