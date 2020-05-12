using Test
using AutoDiff.Backward

@testset "Arithmetic / Algebra" begin
    @testset "f(x,y,z) = x + 2*y/z" begin
        f(x, y, z) = x + 2.0*y/z
        g = gradient(f)

        @test all(isapprox.(g(1.0, 2.0, 3.0), [1.0, 2.0/3.0, -4.0/9.0]))
    end

    @testset "reuse variable" begin
        f(x, y, z) = x - 3.0*x/(y*z) + z*y
        g = gradient(f)

        x = randn()
        y = randn()
        z = randn()

        @test all(isapprox.(g(x, y, z), [1.0 - 3.0/(y*z), 3.0*x/(y*y*z) + z, 3.0*x/(y*z*z) + y]))
    end

    @testset "unary minus" begin
        f(x,y,z) = -x + (-y)*z
        g = gradient(f)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), [-1.0, -z, -y]))
    end
end
