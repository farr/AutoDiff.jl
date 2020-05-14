using Test

@testset "Backward.jl" begin
    include("BackwardTest.jl")
end

@testset "Hierarchical Gaussian Gradient" begin
    include("HierGaussLikelihood.jl")
end
