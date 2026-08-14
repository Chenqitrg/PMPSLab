using Test
import TensorKit
using PMPSLab:
    AbstractPeriodicProduct,
    FormalIdentity,
    UnitalElement,
    edge_space,
    prefix_products,
    suffix_products
using TensorKit: SpaceMismatch, codomain, domain

mutable struct ScanElement
    factors::Vector{Int}
    output_space::Symbol
    input_space::Symbol
end

ScanElement(factor::Int) = ScanElement([factor], :V, :V)

TensorKit.codomain(element::ScanElement) = element.output_space
TensorKit.domain(element::ScanElement) = element.input_space

Base.copy(element::ScanElement) = ScanElement(
    copy(element.factors), element.output_space, element.input_space
)

function Base.:*(left::ScanElement, right::ScanElement)
    domain(left) == codomain(right) || throw(SpaceMismatch())
    return ScanElement(
        vcat(left.factors, right.factors),
        codomain(left),
        domain(right),
    )
end

struct ScanProduct <: AbstractPeriodicProduct{ScanElement}
    elements::Vector{ScanElement}
end

Base.length(product::ScanProduct) = length(product.elements)
Base.getindex(product::ScanProduct, site::Int) = product.elements[site]

@testset "periodic product scans" begin
    product = ScanProduct([ScanElement(site) for site in 1:5])
    prefixes = prefix_products(product)
    suffixes = suffix_products(product)

    @test eltype(prefixes) == UnitalElement{ScanElement,Symbol}
    @test eltype(suffixes) == eltype(prefixes)
    @test prefixes[1] == FormalIdentity{ScanElement}(edge_space(product))
    @test suffixes[end] == FormalIdentity{ScanElement}(edge_space(product))
    @test prefixes[2] !== product[1]
    @test suffixes[end - 1] !== product[length(product)]

    for site in 2:length(product)
        @test (prefixes[site]::ScanElement).factors == collect(1:site-1)
    end
    for site in 1:length(product)-1
        @test (suffixes[site]::ScanElement).factors ==
              collect(site + 1:length(product))
    end

    singleton = ScanProduct([ScanElement(1)])
    @test only(prefix_products(singleton)) isa FormalIdentity
    @test only(suffix_products(singleton)) isa FormalIdentity

    empty = ScanProduct(ScanElement[])
    @test_throws ArgumentError prefix_products(empty)
    @test_throws ArgumentError suffix_products(empty)
end
