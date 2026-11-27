@testset "PMPS owns numerical state and implements A.I." begin
    @test parentmodule(PeriodicMPS) === PMPSLab
    @test parentmodule(PMPSLab.SplitSnapshot) === PMPSLab
    @test PMPSLab.AbstractSplit <: AI.Algorithm
    @test PMPSLab.AbstractSplitState <: AI.State
    @test PMPSLab.split === Base.split
    @test !isdefined(PMPSLab, :ExperimentLab)
    @test !isdefined(PMPSLab, :algorithm_steps)
    @test !isdefined(PMPSLab, :start_run)
end
