using Test
import TensorKit
using PMPSLab: FormalIdentity, UnitalElement, compose_product
using TensorKit: SpaceMismatch, codomain, domain

mutable struct CompositionCounter
    count::Int
end

struct CompositionElement
    value::Int
    counter::CompositionCounter
    output_space::Symbol
    input_space::Symbol
end

CompositionElement(value::Int, counter::CompositionCounter) =
    CompositionElement(value, counter, :V, :V)

TensorKit.codomain(element::CompositionElement) = element.output_space
TensorKit.domain(element::CompositionElement) = element.input_space

function Base.:*(left::CompositionElement, right::CompositionElement)
    left.counter === right.counter || throw(ArgumentError(
        "elements must share one counter"
    ))
    domain(left) == codomain(right) || throw(SpaceMismatch())
    left.counter.count += 1
    return CompositionElement(
        left.value * right.value,
        left.counter,
        codomain(left),
        domain(right),
    )
end

@testset "FormalIdentity — unital composition" begin
    counter = CompositionCounter(0)
    element = CompositionElement(2, counter)
    formal_identity = FormalIdentity{CompositionElement}(:V)

    @test formal_identity isa UnitalElement{CompositionElement,Symbol}
    @test compose_product(formal_identity, element) === element
    @test compose_product(element, formal_identity) === element
    @test compose_product(formal_identity, formal_identity) === formal_identity
    @test formal_identity.space == :V

    product = compose_product(element, element)
    @test product.value == 4
    @test counter.count == 1

    wrong = FormalIdentity{CompositionElement}(:W)
    @test_throws SpaceMismatch compose_product(wrong, element)
    @test_throws SpaceMismatch compose_product(element, wrong)
    @test_throws SpaceMismatch compose_product(formal_identity, wrong)
    @test_throws MethodError compose_product(
        FormalIdentity{Int}(:V), element
    )
end
