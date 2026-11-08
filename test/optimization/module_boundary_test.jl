using PMPSLab: RandomInitialization, IdentityInitialization, SVDInitialization, GaugeInitialization, SplitEnvironmentPolicy, NuclearNorm, NoStopping, CostTolerance, CostConvergence
@testset "public names follow their owning packages" begin
    @test Base.isexported(PMPSLab, :ALSSplit)
    @test Base.isexported(PMPSLab, :split)
    for name in (:SVDInitialization, :NuclearNorm, :DenseTransferMatrix, :TruncatedTransferMatrix)
        @test Base.ispublic(PMPSLab, name)
        @test !Base.isexported(PMPSLab, name)
    end
    for name in (:AbstractInitialization,
                 :AbstractStoppingCriterion, :check_convergence,
                 :SplitState, :initialize_split,
                 :SplitProgress)
        @test Base.ispublic(PMPSLab, name)
        @test !Base.isexported(PMPSLab, name)
    end
    for name in (:AlgorithmContext, :SplitContext, :update!)
        @test !Base.ispublic(PMPSLab, name)
    end
    for name in (:start_run, :build_cache,
                 :AbstractAlgorithmConfig, Symbol("@set"), Symbol("@reset"))
        @test !Base.isexported(PMPSLab, name)
    end
    @test parentmodule(ALSSplit) === PMPSLab
    @test !isdefined(PMPSLab, :start_run)
    @test !isdefined(PMPSLab, :build_cache)
end

@testset "traversal and linear solves have independent owners" begin
    @test parentmodule(CircularSweep) === LatticeIteratorLab
    @test parentmodule(sweep_states) === LatticeIteratorLab
    @test parentmodule(completes_sweep) === LatticeIteratorLab
    @test parentmodule(CholeskySolve) === TensorLinearSolveLab
    for name in (:CircularSweep, :ZigzagSweep, :LUSolve, :LDLTSolve,
                 :CholeskySolve, :KrylovSolve, :AbstractSweep, :AbstractSolve,
                 :sweep_states, :completes_sweep, :ldiv, :rdiv)
        @test !Base.ispublic(PMPSLab, name)
        @test !Base.isexported(PMPSLab, name)
    end
end
