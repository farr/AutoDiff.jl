module StatsFunctions

export logsumexp

function logsumexp(a, b)
    x = max(a,b)
    y = min(a,b)

    x + log1p(exp(y-x))
end

function logsumexp(xs::Array{T}) where T <: Number
    xm = maximum(xs)
    xm + log(sum(exp.(xs.-xm)))
end

end
