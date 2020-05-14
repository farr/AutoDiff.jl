module Backward

export value_and_gradient, gradient

import Base: +, *, -, /, exp, sqrt, log, log1p
import Base: sin, cos, tan, sec, csc, cot
import Base: atan
import Base: sum
import Base: convert, promote_rule, zero, one

import AutoDiff.StatsFunctions: logsumexp

abstract type BADNode{T<:Number} <: Number end

function convert(::Type{BADNode{T}}, x::T) where {T <: Number}
    BADNodeConst(x, zero(T))
end

function promote_rule(::Type{<:BADNode{R}}, ::Type{T}) where {T <: Number, R <: Number}
    BADNode{promote_type(R,T)}
end

function reset!(n::BADNode{T}) where T <: Number
    n.adj = zero(T)
end

mutable struct BADNodeConst{T<:Number} <: BADNode{T}
    value::T
    adj::T
end

function zero(n::BADNode{T}) where T <: Number
    BADNodeConst(zero(T), zero(T))
end

function one(n::BADNode{T}) where T <: Number
    BADNodeConst(one(T), zero(T))
end

function backprop!(n::BADNodeConst{T}) where T <: Number
    # noop---there is nothing to pass upward!
end

function parents(n::BADNodeConst{T}) where T <: Number
    # no parents, so empty list
    BADNode{T}[]
end

mutable struct BADNodePlus{T<:Number} <: BADNode{T}
    value::T
    adj::T
    lparent::BADNode{T}
    rparent::BADNode{T}
end

function backprop!(n::BADNodePlus{T}) where T <: Number
    n.lparent.adj += n.adj
    n.rparent.adj += n.adj
end

function parents(n::BADNodePlus{T}) where T <: Number
    BADNode{T}[n.lparent, n.rparent]
end

function +(a::BADNode{T}, b::BADNode{T}) where T <: Number
    BADNodePlus(a.value + b.value, zero(a.value), a, b)
end

mutable struct BADNodeMinus{T<:Number} <: BADNode{T}
    value::T
    adj::T
    lparent::BADNode{T}
    rparent::BADNode{T}
end

function backprop!(n::BADNodeMinus{T}) where T <: Number
    n.lparent.adj += n.adj
    n.rparent.adj -= n.adj
end

function parents(n::BADNodeMinus{T}) where T <: Number
    BADNode{T}[n.lparent, n.rparent]
end

function -(a::BADNode{T}, b::BADNode{T}) where T <: Number
    BADNodeMinus(a.value - b.value, zero(T), a, b)
end

mutable struct BADNodeUMinus{T<:Number} <: BADNode{T}
    value::T
    adj::T
    parent::BADNode{T}
end

function backprop!(n::BADNodeUMinus{T}) where T <: Number
    n.parent.adj -= n.adj
end

function parents(n::BADNodeUMinus{T}) where T <: Number
    BADNode{T}[n.parent]
end

function -(a::BADNode{T}) where T <: Number
    BADNodeUMinus(-a.value, zero(T), a)
end

mutable struct BADNodeTimes{T<:Number} <: BADNode{T}
    value::T
    adj::T
    lparent::BADNode{T}
    rparent::BADNode{T}
end

function backprop!(n::BADNodeTimes{T}) where T <: Number
    n.lparent.adj += n.adj*n.rparent.value
    n.rparent.adj += n.adj*n.lparent.value
end

function parents(n::BADNodeTimes{T}) where T <: Number
    BADNode{T}[n.lparent, n.rparent]
end

function *(a::BADNode{T}, b::BADNode{T}) where T <: Number
    BADNodeTimes(a.value * b.value, zero(T), a, b)
end

mutable struct BADNodeDiv{T<:Number} <: BADNode{T}
    value::T
    adj::T
    lparent::BADNode{T}
    rparent::BADNode{T}
end

function backprop!(n::BADNodeDiv{T}) where T <: Number
    n.lparent.adj += n.adj / n.rparent.value
    n.rparent.adj -= n.adj * n.value / n.rparent.value
end

function parents(n::BADNodeDiv{T}) where T <: Number
    BADNode{T}[n.lparent, n.rparent]
end

function /(a::BADNode{T}, b::BADNode{T}) where T <: Number
    BADNodeDiv(a.value / b.value, zero(T), a, b)
end

mutable struct BADNodeExp{T<:Number} <: BADNode{T}
    value::T
    adj::T
    parent::BADNode{T}
end

function backprop!(n::BADNodeExp{T}) where T <: Number
    n.parent.adj += n.adj * n.value
end

function parents(n::BADNodeExp{T}) where T <: Number
    BADNode{T}[n.parent]
end

function exp(n::BADNode{T}) where T <: Number
    BADNodeExp(exp(n.value), zero(T), n)
end

mutable struct BADNodeSqrt{T<:Number} <: BADNode{T}
    value::T
    adj::T
    parent::BADNode{T}
end

function backprop!(n::BADNodeSqrt{T}) where T <: Number
    n.parent.adj += n.adj / (2.0*n.value)
end

function parents(n::BADNodeSqrt{T}) where T <: Number
    BADNode{T}[n.parent]
end

function sqrt(n::BADNode{T}) where T <: Number
    BADNodeSqrt(sqrt(n.value), zero(T), n)
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
        mutable struct $nodename{T<:Number} <: BADNode{T}
            x::T
            value::T
            adj::T
            parent::BADNode{T}
        end

        function backprop!(n::$nodename{T}) where T<:Number
            n.parent.adj += n.adj * $df(n.x, n.value)
        end

        function parents(n::$nodename{T}) where T <: Number
            BADNode{T}[n.parent]
        end

        function $f(x::BADNode{T}) where T <: Number
            $nodename(x.value, $f(x.value), zero(T), x)
        end
    end
end

mutable struct BADNodeAtan2{T<:Number} <: BADNode{T}
    x::T
    y::T
    value::T
    adj::T
    lparent::BADNode{T}
    rparent::BADNode{T}
end

function backprop!(n::BADNodeAtan2{T}) where T<:Number
    denom = n.x*n.x + n.y*n.y
    n.lparent.adj += n.adj * n.x / denom
    n.rparent.adj += -n.adj * n.y / denom
end

function parents(n::BADNodeAtan2{T}) where {T<:Number}
    BADNode{T}[n.lparent, n.rparent]
end

function atan(y::BADNode{T}, x::BADNode{T}) where {T<:Number}
    BADNodeAtan2(x.value, y.value, atan(y.value, x.value), zero(T), y, x)
end

mutable struct BADNodeSum{T<:Number} <: BADNode{T}
    value::T
    adj::T
    parents::Array{<:BADNode{T}}
end

function backprop!(n::BADNodeSum{T}) where T<:Number
    for p in n.parents
        p.adj += n.adj
    end
end

function parents(n::BADNodeSum{T}) where T<:Number
    n.parents.reshape((prod(size(n.parents)...),))
end

function sum(xs::Array{<:BADNode{T}, N}) where {T<:Number, N<:Integer}
    BADNodeSum(sum([x.value for x in xs]), zero(xs[1].value), xs)
end

mutable struct BADNodeLog{T<:Number} <: BADNode{T}
    x::T
    value::T
    adj::T
    parent::BADNode{T}
end

function backprop!(n::BADNodeLog{T}) where T <: Number
    n.parent.adj += n.adj / n.x
end

function parents(n::BADNodeLog{T}) where T <: Number
    BADNode{T}[n.parent]
end

function log(n::BADNode{T}) where T <: Number
    BADNodeLog(n.value, log(n.value), zero(T), n)
end

mutable struct BADNodeLog1p{T<:Number} <: BADNode{T}
    x::T
    value::T
    adj::T
    parent::BADNode{T}
end

function backprop!(n::BADNodeLog1p{T}) where T <: Number
    n.parent.adj += n.adj / (one(T) + n.x)
end

function parents(n::BADNodeLog1p{T}) where T <: Number
    BADNode{T}[n.parent]
end

function log1p(n::BADNode{T}) where T <: Number
    BADNodeLog1p(n.value, log1p(n.value), zero(T), n)
end

function breadth_first_apply!(fn, bplist)
    while length(bplist) > 0
        n = popfirst!(bplist)
        fn(n)
        push!(bplist, parents(n)...)
    end
end

function value_and_gradient(f)
    function vgf(x::T) where T<:Number
        y = convert(BADNode{T}, x)
        r = f(y)
        r.adj = one(T)
        bplist = BADNode{T}[r]
        breadth_first_apply!(backprop!, bplist)
        (r.value, r.adj)
    end

    function vgf(x::Vararg{T,N}) where {T<:Number, N}
        y = BADNode{T}[convert(BADNode{T}, xx) for xx in x]
        r = f(y...)
        r.adj = one(T)
        bplist = BADNode{T}[r]
        breadth_first_apply!(backprop!, bplist)

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
