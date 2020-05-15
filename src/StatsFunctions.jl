module StatsFunctions

export logsumexp

function logsumexp(a, b)
    x = max(a,b)
    y = min(a,b)

    x + log1p(exp(y-x))
end

function logsumexp(xs::Array{T,N}) where {T <: Number, N}
    xm = maximum(xs)
    xm + log(sum(exp.(xs.-xm)))
end

end
