module Backward

export value_and_gradient, gradient

import Base: ==, !=, <, <=, >, >=, isless, isequal
import Base: +, *, -, /, exp, sqrt, log, log1p
import Base: sin, cos, tan, sec, csc, cot
import Base: atan
import Base: sum
import Base: convert, promote_rule, zero, one

import AutoDiff.StatsFunctions: logsumexp

mutable struct BADNode{T<:Number} <: Number
    tape::Array{BADNode{T},1}
    value::T
    adj::T
    backprop!
end

function zero(::Type{BADNode{T}}) where {T <: Number}
    BADNode(BADNode{T}[], zero(T), zero(T), (n)->())
end

function zero(x::BADNode{T}) where {T <: Number}
    BADNode(BADNode{T}[], zero(T), zero(T), (n)->())
end

function one(::Type{BADNode{T}}) where {T <: Number}
    BADNode(BADNode{T}[], one(T), zero(T), (n)->())
end

function one(x::BADNode{T}) where {T <: Number}
    BADNode(BADNode{T}[], one(T), zero(T), (n)->())
end

function convert(::Type{BADNode{T}}, x::T) where {T <: Number}
    BADNode(BADNode{T}[], x, zero(T), (n)->())
end

function promote_rule(::Type{BADNode{T}}, ::Type{T}) where {T <: Number}
    BADNode{T}
end

for fn in [Symbol("=="), Symbol("!="), Symbol("<"), Symbol("<="), Symbol(">"), Symbol(">="), :isless, :isequal]
    @eval begin
        $fn(a::BADNode{T}, b::BADNode{T}) where {T <: Number} = $fn(a.value, b.value)
    end
end

function +(x::BADNode{T}, y::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    function bp!(n::BADNode{T})
        x.adj += n.adj
        y.adj += n.adj
    end
    n = BADNode(t, x.value + y.value, zero(T), bp!)
    push!(t, n)
    n
end

function -(x::BADNode{T}, y::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    function bp!(n::BADNode{T})
        x.adj += n.adj
        y.adj -= n.adj
    end
    n = BADNode(t, x.value - y.value, zero(T), bp!)
    push!(t, n)
    n
end

function -(x::BADNode{T}) where {T <: Number}
    t = x.tape
    function bp!(n::BADNode{T})
        x.adj -= n.adj
    end
    n = BADNode(t, -x.value, zero(T), bp!)
    push!(t, n)
    n
end

function *(x::BADNode{T}, y::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    function bp!(n::BADNode{T})
        x.adj += y.value*n.adj
        y.adj += x.value*n.adj
    end
    n = BADNode(t, x.value*y.value, zero(T), bp!)
    push!(t, n)
    n
end

function /(x::BADNode{T}, y::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    r = x.value / y.value
    function bp!(n::BADNode{T})
        x.adj += n.adj / y.value
        y.adj -= r / y.value * n.adj
    end
    n = BADNode(t, r, zero(T), bp!)
    push!(t, n)
    n
end

function exp(x::BADNode{T}) where {T <: Number}
    t = x.tape
    r = exp(x.value)
    function bp!(n::BADNode{T})
        x.adj += r*n.adj
    end
    n = BADNode(t, r, zero(T), bp!)
    push!(t, n)
    n
end

function sqrt(x::BADNode{T}) where {T <: Number}
    t = x.tape
    r = sqrt(x.value)
    function bp!(n::BADNode{T})
        x.adj += n.adj / (convert(T, 2)*r)
    end
    n = BADNode(t, r, zero(T), bp!)
    push!(t, n)
    n
end

function sin_deriv(x,y)
    cos(x)
end

function cos_deriv(x,y)
    -sin(x)
end

function tan_deriv(x, y)
    z = sec(x)
    z*z
end

function sec_deriv(x, y)
    y*tan(x)
end

function csc_deriv(x, y)
    -y*cot(x)
end

function cot_deriv(x, y)
    z = csc(x)
    -z*z
end

function atan_deriv(x, y)
    one(x) / (one(x) + x*x)
end

for (f, df) in zip([:sin, :cos, :tan, :sec, :csc, :cot, :atan], [:sin_deriv, :cos_deriv, :tan_deriv, :sec_deriv, :csc_deriv, :cot_deriv, :atan_deriv])
    fstr = uppercasefirst(string(f))
    nodename = Symbol("BADNode$(fstr)")
    @eval begin
        function $f(x::BADNode{T}) where {T <: Number}
            t = x.tape
            r = $f(x.value)
            function bp!(n::BADNode{T})
                x.adj += $df(x.value, r)*n.adj
            end
            n = BADNode(t, r, zero(T), bp!)
            push!(t, n)
            n
        end
    end
end

function atan(y::BADNode{T}, x::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    function bp!(n::BADNode{T})
        denom = x.value*x.value + y.value*y.value
        y.adj += n.adj * x.value / denom
        x.adj -= n.adj * y.value / denom
    end
    n = BADNode(t, atan(y.value, x.value), zero(T), bp!)
    push!(t, n)
    n
end

function sum(x::Array{BADNode{T}, N}) where {T <: Number, N}
    t = BADNode{T}[]
    for i in eachindex(x)
        if length(x[i].tape) > 0
            t = x[i].tape
            break
        end
    end

    s = zero(T)
    for i in eachindex(x)
        s += x[i].value
    end

    function bp!(n::BADNode{T})
        for i in eachindex(x)
            x[i].adj += n.adj
        end
    end

    n = BADNode(t, s, zero(T), bp!)
    push!(t, n)
    n
end

function log(x::BADNode{T}) where {T <: Number}
    t = x.tape
    function bp!(n::BADNode{T})
        x.adj += n.adj / x.value
    end
    n = BADNode(t, log(x.value), zero(T), bp!)
    push!(t, n)
    n
end

function log1p(x::BADNode{T}) where {T <: Number}
    function bp!(n::BADNode{T})
        x.adj += n.adj / (one(T) + x.value)
    end
    n = BADNode(x.tape, log1p(x.value), zero(T), bp!)
    push!(x.tape, n)
    n
end

function logsumexp(x::BADNode{T}, y::BADNode{T}) where {T <: Number}
    t = (length(x.tape) == 0 ? y.tape : x.tape)
    r = logsumexp(x.value, y.value)
    function bp!(n::BADNode{T})
        x.adj += n.adj * exp(x.value - r)
        y.adj += n.adj * exp(y.value - r)
    end
    n = BADNode(t, r, zero(T), bp!)
    push!(t, n)
    n
end

function logsumexp(x::Array{BADNode{T}, N}) where {T <: Number, N}
    t = BADNode{T}[]
    for i in eachindex(x)
        if length(x[i].tape) > 0
            t = x[i].tape
            break
        end
    end

    xx = Array{T, N}(undef, size(x))
    for i in eachindex(x)
        xx[i] = x[i].value
    end
    r = logsumexp(xx)

    function bp!(n::BADNode{T})
        for i in eachindex(x)
            x[i].adj += n.adj*exp(x[i].value - r)
        end
    end

    n = BADNode(t, r, zero(T), bp!)
    push!(t, n)
    n
end

function value_and_gradient(f)
    function vgf(x::T) where T<:Number
        t = BADNode{T}[]
        y = BADNode(t, x, zero(T), (n)->())
        push!(t, y)
        r = f(y)
        r.adj = one(T)
        while length(t) > 0
            n = pop!(t)
            n.backprop!(n)
        end
        (r.value, y.adj)
    end

    function vgf(xs::Array{T, N}) where {T <: Number, N}
        t = BADNode{T}[]
        ys = Array{BADNode{T}, N}(undef, size(xs))
        for i in eachindex(ys)
            ys[i] = BADNode(t, xs[i], zero(T), (n)->())
            push!(t, ys[i])
        end
        r = f(ys)
        r.adj = one(T)

        while length(t) > 0
            n = pop!(t)
            n.backprop!(n)
        end

        zs = Array{T, N}(undef, size(xs))
        for i in eachindex(xs)
            zs[i] = ys[i].adj
        end
        (r.value, zs)
    end

    function vgf(x::Vararg{T,N}) where {T<:Number, N}
        t = BADNode{T}[]

        y = Array{BADNode{T}, 1}(undef, length(x))
        for i in eachindex(x)
            y[i] = BADNode(t, x[i], zero(T), (n)->())
            push!(t, y[i])
        end

        r = f(y...)
        r.adj = one(T)

        while length(t) > 0
            n = pop!(t)
            n.backprop!(n)
        end

        g = T[n.adj for n in y]
        (r.value, g)
    end

    vgf
end

function gradient(f)
    vgf = value_and_gradient(f)
    function g(x...)
        vgf(x...)[2]
    end

    g
end

end
