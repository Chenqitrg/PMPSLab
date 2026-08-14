using Test
import TensorKit
using PMPSLab: AbstractPeriodicProduct, FormalIdentity, edge_identity, edge_space
using TensorKit: SpaceMismatch, Z2Space, ℂ, codomain, domain

struct EdgeElement{S}
    output_space::S
    input_space::S
end

TensorKit.codomain(element::EdgeElement) = element.output_space
TensorKit.domain(element::EdgeElement) = element.input_space

struct EdgeProduct{T} <: AbstractPeriodicProduct{T}
    elements::Vector{T}
end

Base.length(product::EdgeProduct) = length(product.elements)
Base.getindex(product::EdgeProduct, site::Int) = product.elements[site]

@testset "AbstractPeriodicProduct — edge space" begin
    complex_space = ℂ^3
    product = EdgeProduct([
        EdgeElement(complex_space, ℂ^2),
        EdgeElement(ℂ^4, complex_space),
    ])
    @test edge_space(product) == complex_space
    @test edge_identity(product) ==
          FormalIdentity{eltype(product)}(complex_space)

    graded_space = Z2Space(0 => 2, 1 => 1)
    graded = EdgeProduct([EdgeElement(graded_space, graded_space)])
    @test edge_space(graded) == graded_space
    @test edge_identity(graded) ==
          FormalIdentity{eltype(graded)}(graded_space)

    open_product = EdgeProduct([
        EdgeElement(ℂ^2, ℂ^2),
        EdgeElement(ℂ^2, ℂ^3),
    ])
    @test_throws SpaceMismatch edge_space(open_product)

    empty = EdgeProduct(EdgeElement{typeof(complex_space)}[])
    @test_throws ArgumentError edge_space(empty)
end
