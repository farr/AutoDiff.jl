module HierGaussLikelihood

using AutoDiff.Backward

# This is a standard hierarchical likelihood.  We imagine that we have a set of
# objects, each of which has a parameter `x`.  Collectively, the `x` are
# distributed according to a Gaussian with mean `mu` and s.d. `sigma`.  We have
# some number of observations of each x, `xobs` with a known uncertainty
# (assumed Gaussian).  We want to be able to take gradients with respect to
# `mu`, `sigma`, and the `x`.

function make_hglikelihood(xobs, sigmaobs)
    nx, no = size(xobs)

    function loglike(mu, sigma, xs...)
        ll = sum(-0.5.*(xs.-mu).*(xs.-mu)./(sigma*sigma)) - float(nx)*log(sigma)

        ll2 = sum(-0.5.*(xobs .- xs).*(xobs .- mu)./(sigmaobs.*sigmaobs) .- log.(sigmaobs))

        ll + ll2
    end

    loglike
end

nx = 100
no = 5

mu = 0.0
sigma = 1.0

xs = mu .+ sigma.*randn(nx)
sigmaobs = 1.0 .+ 1.0.*rand(nx, no)
xobs = xs .+ sigmaobs.*randn(nx, no)

loglike = make_hglikelihood(xobs, sigmaobs)
g = gradient(loglike)

g(mu, sigma, xs...)

end
