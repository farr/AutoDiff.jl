using Test
using AutoDiff.Backward

@testset "f(x,y,z) = x + 2*y/z" begin
    f(x, y, z) = x + 2.0*y/z
    g = gradient(f)

    @test all(isapprox.(g(1.0, 2.0, 3.0), [1.0, 2.0/3.0, -4.0/9.0]))
end

@testset "one-arg functions" begin
    f(x) = 2.0*x + 3.0
    g = gradient(f)

    @test isapprox(g(1.0), 2.0)
end

@testset "vector-mode functions" begin
    f(xs) = sqrt(xs[1]*xs[1] + xs[2]*xs[2] + xs[3]*xs[3])
    g = gradient(f)
    xs = randn(3)
    @test all(isapprox.(g(xs), [xs[1]/f(xs), xs[2]/f(xs), xs[3]/f(xs)]))
end

@testset "array-mode functions" begin
    f(xs) = sqrt(sum(xs.*xs))
    g = gradient(f)
    xs = randn(3, 5, 7)
    r = f(xs)
    @test all(isapprox.(g(xs), xs./r))
    @test size(g(xs))==(3,5,7)
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

@testset "exp" begin
    f(x,y,z) = x*exp(y*z)
    g = gradient(f)
    x,y,z = randn(3)
    ff = f(x,y,z)
    @test all(isapprox.(g(x,y,z), [ff/x, z*ff, y*ff]))
end

@testset "sqrt" begin
    f(x,y,z) = sqrt(x*x + y*y + z*z)
    f_and_g = value_and_gradient(f)
    x,y,z = randn(3)

    ff, gg = f_and_g(x,y,z)

    @test all(isapprox.(gg, [x/ff, y/ff, z/ff]))
end

@testset "trig" begin
    @testset "sin" begin
        f(x,y,z) = x*sin(y*z)
        g = gradient(f)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), [sin(y*z), x*z*cos(y*z), x*y*cos(y*z)]))
    end

    @testset "cos" begin
        f(x,y,z) = x*cos(y*z)
        g = gradient(f)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), [cos(y*z), -z*x*sin(y*z), -x*y*sin(y*z)]))
    end

    @testset "tan" begin
        f(x,y,z) = x*tan(y*z)
        ff(x,y,z) = x*sin(y*z)/cos(y*z)
        g = gradient(f)
        gg = gradient(ff)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), gg(x,y,z)))
    end

    @testset "sec" begin
        f(x,y,z) = x*sec(y*z)
        ff(x,y,z) = x/cos(y*z)
        g = gradient(f)
        gg = gradient(ff)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), gg(x,y,z)))
    end

    @testset "csc" begin
        f(x,y,z) = x*csc(y*z)
        ff(x,y,z) = x/sin(y*z)
        g = gradient(f)
        gg = gradient(ff)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), gg(x,y,z)))
    end

    @testset "cot" begin
        f(x,y,z) = x*cot(y*z)
        ff(x,y,z) = x/tan(y*z)
        g = gradient(f)
        gg = gradient(ff)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), gg(x,y,z)))
    end

    @testset "atan" begin
        f(x,y,z) = atan(fff(x,y,z))
        ff(x,y,z) = tan(f(x,y,z))
        fff(x,y,z) = sqrt(x*x+y*y+z*z)
        g = gradient(f)
        gg = gradient(ff)
        ggg = gradient(fff)
        x,y,z = randn(3)
        @test all(isapprox.(ggg(x,y,z), gg(x,y,z)))
    end

    @testset "atan2" begin
        f(theta) = atan(sin(theta),cos(theta))
        g = gradient(f)
        theta = 2*pi*randn()
        @test isapprox(g(theta), 1.0)
    end
end

@testset "sum like functions" begin
    @testset "sum" begin
        f(x, y, z) = sum([x,y,z])
        g = gradient(f)
        x,y,z = randn(3)
        @test all(isapprox.(g(x,y,z), ones(3)))
    end
end

@testset "log" begin
    f(x,y,z) = log(x*y*z)
    g = gradient(f)
    x,y,z = abs.(randn(3))
    @test all(isapprox.(g(x,y,z), [1.0/x, 1.0/y, 1.0/z]))
end

@testset "log1p" begin
    f(x,y,z) = log1p(x*y*z)
    g = gradient(f)
    x,y,z = 1e-3.*randn(3)
    @test all(isapprox.(g(x,y,z), [y*z/(1+x*y*z), x*z/(1+x*y*z), x*y/(1+x*y*z)]))
end

@testset "logsumexp" begin
    f(x,y,z) = logsumexp(x, logsumexp(y, z))
    f_exact(x,y,z) = log(exp(x) + exp(y) + exp(z))
    g = gradient(f)
    g_exact = gradient(f_exact)
    gg = gradient(logsumexp) # Array version

    x,y,z = randn(3)

    @test all(isapprox.(g(x,y,z), g_exact(x,y,z)))
    @test all(isapprox.(gg([x,y,z]), g_exact(x,y,z)))
end
