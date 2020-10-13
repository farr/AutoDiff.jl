A good, clean Julia autodiff package.

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

Performance is on par with the best autodiff frameworks out there.
