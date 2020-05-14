using Test

@testset "StatsFunctions.jl" begin
    include("StatsFunctionsTest.jl")
end

@testset "Backward.jl" begin
    include("BackwardTest.jl")
end

@testset "Hierarchical Gaussian Gradient" begin
    include("HierGaussLikelihood.jl")
end
