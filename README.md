A good, clean Julia autodiff package.

To install into your existing Julia code, enter the following at the Julia REPL after activating your environment:

```juliarepl
julia> using Pkg
julia> pkg"add https://github.com/farr/AutoDiff.jl.git"
```

To use:

```julia
using AutoDiff.Backward

function rosenbrock(x, y)
  a = 1.0
  b = 100.0
  term1 = (a - x)
  term2 = (y - x*x)

  return a*term1*term1 + b*term2*term2
end

rgrad = gradient(rosenbrock)

println("Gradient at minimum = $(rgrad(1.0, 1.0))") # [0.0, 0.0]
println("Gradient at origin = $(rgrad(0.0, 0.0))") # [-2.0, 0.0]
```

Arithmetic, trig functions, exponential functions, and some simple statistical
functions (`logsumexp`) and any composition/calculation on these work; but
currently no linear algebra.  This package is a WIP.

Performance is on par with the best autodiff frameworks out there.  For example, the cost to take the gradient of a simple hierarchical Gaussian likelihood (see, e.g., [the eight schools problem](https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/)) with more than 1000 parameters is only an order of magnitude more that the computation of the log likelihood:

```julia-repl
(@v1.5) pkg> activate .
 Activating environment at `~/Code/AutoDiff/Project.toml`

julia> using BenchmarkTools

julia> include("test/HierGaussLikelihood.jl")
Main.HierGaussLikelihood

julia> @benchmark HierGaussLikelihood.loglike(HierGaussLikelihood.mu, HierGaussLikelihood.sigma, HierGaussLikelihood.xs...)
BenchmarkTools.Trial:
  memory estimate:  276.36 KiB
  allocs estimate:  3043
  --------------
  minimum time:     150.312 μs (0.00% GC)
  median time:      177.813 μs (0.00% GC)
  mean time:        209.936 μs (8.92% GC)
  maximum time:     6.700 ms (95.49% GC)
  --------------
  samples:          10000
  evals/sample:     1

julia> @benchmark HierGaussLikelihood.g(HierGaussLikelihood.mu, HierGaussLikelihood.sigma, HierGaussLikelihood.xs...)
BenchmarkTools.Trial:
  memory estimate:  7.66 MiB
  allocs estimate:  162080
  --------------
  minimum time:     2.548 ms (0.00% GC)
  median time:      2.909 ms (0.00% GC)
  mean time:        3.724 ms (18.68% GC)
  maximum time:     14.017 ms (60.44% GC)
  --------------
  samples:          1342
  evals/sample:     1

julia>
```

The implementation here is pretty similar to the Stan Math Library in spirit ([Carpenter, et al. (2015)](https://arxiv.org/abs/1509.07164)), though we exploit some Julia features not available in C++.
