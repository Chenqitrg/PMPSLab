struct TestMeasurement <: AbstractMeasurement end

mutable struct TestMeasurementResult <: AbstractMeasurementResult
    label::Symbol
    payload::Vector{Int}
end

PMPSLab.measure(::TestMeasurement, label::Symbol, value::Int) =
    TestMeasurementResult(label, [value])

@testset "generic measurement interface" begin
    result = measure(TestMeasurement(), :rg0, 1)

    @test result isa AbstractMeasurementResult
    @test result.label == :rg0
    @test result.payload == [1]
end

@testset "MeasurementHistory — ordering and copy isolation" begin
    first_result = TestMeasurementResult(:rg0, [1])
    second_result = TestMeasurementResult(:rg1, [2])
    empty_history = MeasurementHistory()

    @test isempty(empty_history)
    @test eltype(empty_history) === AbstractMeasurementResult
    @test push!(empty_history, first_result) === empty_history

    source = TestMeasurementResult[first_result]
    history = MeasurementHistory(source)

    push!(source, second_result)
    @test length(history) == 1

    @test push!(history, second_result) === history
    @test collect(history) == [first_result, second_result]
    @test history[1] === first_result
    @test history[2] === second_result

    copied = copy(history)
    push!(copied, TestMeasurementResult(:rg2, [3]))
    @test length(history) == 2
    @test length(copied) == 3
    @test copied[1] === history[1]

    detached = deepcopy(history)
    push!(detached[1].payload, 10)
    @test detached[1] !== history[1]
    @test detached[1].payload == [1, 10]
    @test history[1].payload == [1]
end

@testset "measurement results stay separate from control logs" begin
    log = SiteLog(:state, 0.25, 0.5)

    @test fieldnames(typeof(log)) == (:state, :cost, :duration)
    @test_throws MethodError SiteLog(
        :state, 0.25, 0.5, TestMeasurementResult(:site, [1])
    )
end
