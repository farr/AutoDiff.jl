module Backward

export value_and_gradient, gradient

import Base: +, *, -, /, exp, convert, promote_rule, zero, one

abstract type BADNode{T<:Number} <: Number end

function convert(::Type{BADNode{T}}, x::T) where T <: Number
    BADNodeConst(x, zero(x))
end

function promote_rule(::Type{S}, ::Type{T}) where {T <: Number, S <: BADNode{T}}
    BADNode{T}
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

function value_and_gradient(f)
    function vgf(x::Vararg{T,N}) where {T<:Number, N}
        y = BADNode{T}[convert(BADNode{T}, xx) for xx in x]
        r = f(y...)
        r.adj = one(T)
        bplist = BADNode{T}[r]

        while length(bplist) > 0
            n = popfirst!(bplist)
            backprop!(n)
            push!(bplist, parents(n)...)
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
