using Test
import LinearAlgebra
import TensorKit
using PMPSLab:
    AbstractPeriodicProduct,
    Backward,
    FormalIdentity,
    Forward,
    SweepState,
    advance!,
    build_cache,
    environment,
    rebuild!
using TensorKit: SpaceMismatch, codomain, domain

mutable struct CacheMultiplicationCounter
    count::Int
end

mutable struct CountedCacheElement
    value::Int
    counter::CacheMultiplicationCounter
    output_space::Symbol
    input_space::Symbol
end

CountedCacheElement(value::Int, counter::CacheMultiplicationCounter) =
    CountedCacheElement(value, counter, :V, :V)

TensorKit.codomain(element::CountedCacheElement) = element.output_space
TensorKit.domain(element::CountedCacheElement) = element.input_space

struct CountedProduct <: AbstractPeriodicProduct{CountedCacheElement}
    factors::Vector{CountedCacheElement}
end

Base.length(product::CountedProduct) = length(product.factors)
Base.getindex(product::CountedProduct, site::Int) = product.factors[site]

Base.copy(element::CountedCacheElement) =
    CountedCacheElement(
        element.value,
        element.counter,
        element.output_space,
        element.input_space,
    )

function Base.copy!(
        destination::CountedCacheElement,
        source::CountedCacheElement,
    )
    codomain(destination) == codomain(source) || throw(SpaceMismatch())
    domain(destination) == domain(source) || throw(SpaceMismatch())
    destination.value = source.value
    return destination
end

function Base.:*(left::CountedCacheElement, right::CountedCacheElement)
    left.counter === right.counter || throw(ArgumentError(
        "elements must share one counter"
    ))
    domain(left) == codomain(right) || throw(SpaceMismatch())
    left.counter.count += 1
    return CountedCacheElement(
        left.value * right.value,
        left.counter,
        codomain(left),
        domain(right),
    )
end

function LinearAlgebra.mul!(
        destination::CountedCacheElement,
        left::CountedCacheElement,
        right::CountedCacheElement,
    )
    product = left * right
    destination.value = product.value
    return destination
end

@testset "PeriodicCache — singleton environment is a formal identity" begin
    counter = CacheMultiplicationCounter(0)
    product = CountedProduct([CountedCacheElement(2, counter)])
    cache = build_cache(product)

    @test environment(cache, 1) == FormalIdentity{CountedCacheElement}(:V)
    @test advance!(
        cache, CountedCacheElement(3, counter), SweepState{Forward}(1, 0)
    ) === cache
    @test advance!(
        cache, CountedCacheElement(4, counter), SweepState{Backward}(1, 0)
    ) === cache
    @test rebuild!(cache, product) === cache
    @test counter.count == 0
end

@testset "PeriodicCache — formal boundaries never multiply" begin
    counter = CacheMultiplicationCounter(0)
    product = CountedProduct([
        CountedCacheElement(site, counter) for site in 1:4
    ])
    cache = build_cache(product)
    @test counter.count == 4

    counter.count = 0
    @test environment(cache, 1).value == 24
    @test environment(cache, 4).value == 6
    @test counter.count == 0

    @test environment(cache, 2).value == 12
    @test counter.count == 1
end
