using Test
using AutoDiff.StatsFunctions

@testset "logsumexp tests" begin
    @testset "two-arg" begin
        x, y = randn(2)
        @test isapprox(logsumexp(x,y), log(exp(x) + exp(y)))
    end

    @testset "array" begin
        xs = randn(32)
        @test isapprox(logsumexp(xs), log(sum(exp.(xs))))
    end
end
