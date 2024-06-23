### Julia implementation of some bivariate copulas
### Author: Arturo Erdely
### Version date: 2024-02-11

copulaList = ["copW", "copΠ", "copH", "copM", 
              "copClayton", "copFrank", "copA4212",
              "copFlip", "copConvex", "copGluing"
]

function copulaIndex()
    for copula ∈ copulaList
        println(copula)
    end
end


## Fréchet-Hoeffding lower bound (W)

function copW()
    tail = true
    archimedean = true
    strict = "never"
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            valor = 1.0 - t
        end
        return valor
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = max(u + v - 1.0, 0.0)
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ 1.0 - u < v ≤ 1.0
            valor = 1.0
        elseif 0.0 ≤ v < 1.0 - u ≤ 1.0
            valor = 0.0
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            valor = 1.0 - u + 0.0*t 
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ 1.0 - u < v ≤ 1.0
            valor = 1.0
        elseif 0.0 ≤ v < 1.0 - u ≤ 1.0
            valor = 0.0
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = 0.0
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int)
        u = rand(n)
        v = 1.0 .- u
        return hcat(u, v)
    end
    return (name = "W", taildep = tail, arch = archimedean, strict = strict, gen = ϕ, 
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC,
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC
    )
end


## Independence or product copula (Π)

function copΠ()
    tail = false
    archimedean = true
    strict = "always"
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            valor = -log(t)
        end
        return valor
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = u * v
        end
        return valor 
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = v
        end
        return valor
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            valor = t + 0.0*u 
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = u
        end
        return valor
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = 1.0
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int)
        u = rand(n)
        v = rand(n)
        return hcat(u, v)
    end
    return (name = "Π", taildep = tail, arch = archimedean, strict = strict, gen = ϕ, 
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC
    )
end


## Copula H(u,v) = uv/(u+v-uv)  <-- particular limiting case of Ali-Mikhail-Haq copulas

function copH()
    tail = false
    archimedean = true
    strict = "always"
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            valor = 1/t - 1
        end
        return valor
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 < u ≤ 1.0 && 0.0 < v ≤ 1.0
            valor = u*v / (u + v - u*v)
        elseif (u == 0.0 && 0.0 ≤ v ≤ 1.0) || (v == 0.0 && 0.0 ≤ u ≤ 1.0)
            valor = 0.0
        end
        return valor 
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (v / (u + v - u*v)) ^ 2
        end
        return valor
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            valor = u*√t / (1 - (1-u)*√t) 
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (u / (u + v - u*v)) ^ 2 
        end
        return valor
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = 2*u*v / (u + v - u*v) ^ 3 
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int) # Nelsen (2006), p.42
        u = rand(n)
        t = rand(n)
        v = @. u*√t / (1.0 - (1.0 - u)*√t)
        return hcat(u, v)
    end
    return (name = "H", taildep = tail, arch = archimedean, strict = strict, gen = ϕ, 
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC
    )
end


## Fréchet-Hoeffding upper bound (M)

function copM()
    tail = true 
    archimedean = false
    C(u::AbstractFloat, v::AbstractFloat) = min(u, v) * (0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0)
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 < u < v < 1.0
            valor = 1.0
        elseif 0.0 < v < u < 1.0
            valor = 0.0
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            valor = u + 0.0*t 
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 < u < v < 1.0
            valor = 0.0
        elseif 0.0 < v < u < 1.0
            valor = 1.0
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = 0.0
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int)
        u = rand(n)
        v = u
        return hcat(u, v)
    end
    return (name = "M", taildep = tail, arch = archimedean, copula = C,
            ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC)
end


## some calculations require numerical approximations using:
"""
    biseccion(f, a, b; δ = abs((a + b)/2) / 1_000_000, m = 10_000)

`f` a continuous real function over the closed interval [a,b] such that `f(a)f(b)<0`
with a maximum approximation error `δ` to a value `c` such that `f(c)=0` in a maximum
number of iterations `m` (bisection method).

## Example
```
f(x) = (x - 3) * (x - 1) * (x + 1)
biseccion(f, -1.9, 0)
```
"""
function biseccion(f, a, b; δ = abs((a + b)/2) / 1_000_000, m = 10_000)
    iter = 1
    z = (a + b) / 2
    while iter ≤ m
        if f(z) == 0.0 || (b - a)/2 ≤ δ
            break # stoping rule
        elseif f(a) * f(z) > 0
            a = z
        else
            b = z
        end
        z = (a + b) / 2
        iter += 1
    end
    iter -= 1
    if iter == m
        @warn "maximum number $m of iterations was reached without a solution"
    end
    return (raiz = z, dif = f(z), numiter = iter, maxiter = m, tol = δ)
end


## CLAYTON #############################

function copClayton(θ::AbstractFloat)
    if θ < -1.0
        error("Parameter must be ≥ -1.0")
        return nothing
    end
    Ω = ("θ ∈ [-1, ∞)", -1.0, Inf)
    tail = true
    archimedean = true
    strict = "just if θ ≥ 0"
    comprehensive = true
    cases = ["W", "Π", "H", "M"]
    paramcases = [-1.0, 0.0, 1.0, Inf]
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            if θ == 0.0
                valor = -log(t)
            else
                valor = (t^(-θ) - 1.0) / θ
            end
        end
        return valor
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 < u ≤ 1.0 && 0.0 < v ≤ 1.0
            w = u^(-θ) + v^(-θ) - 1.0
            if w > 0
                valor = w^(-1.0 / θ)
            else
                valor = 0.0
            end
        elseif (u == 0.0 && 0.0 ≤ v ≤ 1.0) || (v == 0.0 && 0.0 ≤ u ≤ 1.0)
            valor = 0.0
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 < u ≤ 1.0 && 0.0 < v ≤ 1.0
            w = u^(-θ) + v^(-θ) - 1.0
            if w > 0
                valor = (u^(-θ-1)) * w^(-1/θ - 1)
            else
                valor = 0.0
            end
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            if θ == 0.0
                valor = t + 0.0*u
            else
                valor = (1 - (1 - t^(-θ/(θ+1)))*(u^(-θ)))^(-1/θ)
            end
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 < u ≤ 1.0 && 0.0 < v ≤ 1.0
            w = u^(-θ) + v^(-θ) - 1.0
            if w > 0
                valor = (v^(-θ-1)) * w^(-1/θ - 1)
            else
                valor = 0.0
            end
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        valor = NaN
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if θ == 0.0
                valor = 1.0
            else
                w = u^(-θ) + v^(-θ) - 1.0
                if w > 0
                    valor = (θ+1) * ((u*v)^(-θ-1)) * (w^(-1/θ - 2))
                else
                    valor = 0.0
                end 
            end 
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int) # General method, see Nelsen (2006) p.40
        u = rand(n)
        t = rand(n)
        if θ == 0.0
            return hcat(u, t)
        else
            v = @. (1 - (1 - t^(-θ/(θ+1)))*(u^(-θ)))^(-1/θ)
            return hcat(u, v)
        end
    end
    return (name = "Clayton", param = θ, paramspace = Ω, taildep = tail, arch = archimedean,
            strict = strict, gen = ϕ, comp = comprehensive, cases = cases, pcases = paramcases,
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC 
    )
end


## FRANK ##############################

function copFrank(θ::AbstractFloat)
    Ω = ("θ ∈ (-∞, ∞)", -Inf, Inf)
    tail = false
    archimedean = true
    strict = "always"
    comprehensive = true
    cases = ["W", "Π", "M"]
    paramcases = [-Inf, 0.0, Inf]
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            if θ == 0.0
                valor = -log(t)
            else
                valor = -log((exp(-θ*t) - 1.0) / (exp(-θ) - 1.0)) 
            end
        end
        return valor
    end
    function invϕ(z::AbstractFloat)
        valor = NaN
        if z ≥ 0.0
            if θ == 0.0
                valor = exp(-z)
            else
                valor = -log(1.0 + (exp(-θ) - 1.0)/exp(z)) / θ
            end
        end
        return valor
    end
    function dϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 < t < 1.0
            if θ == 0.0
                valor = -1.0 / t
            else
                valor = (θ * exp(-t*θ)) / (exp(-t*θ) - 1.0)
            end
        end
        return valor
    end
    function invdϕ(y::AbstractFloat) 
        valor = NaN
        if y < 0.0
            if θ == 0.0
                valor = -1.0 / y
            else
                valor = log(1.0 - θ/y) / θ
            end
        end 
        return valor 
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = -log(1 + (exp(-θ*u) - 1)*(exp(-θ*v) - 1) / (exp(-θ) - 1)) / θ
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (exp(-u*θ)*(exp(-v*θ)-1))/((exp(-θ)-1)*(((exp(-u*θ)-1)*(exp(-v*θ)-1))/(exp(-θ)-1)+1))
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u) 
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            if θ == 0.0
                valor = t
            else
                valor = log((t*exp(u*θ+θ)+(1-t)*exp(θ))/(t*exp(u*θ)+(1-t)*exp(θ)))/θ
            end
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat) 
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (exp(-v*θ)*(exp(-u*θ)-1))/((exp(-θ)-1)*(((exp(-v*θ)-1)*(exp(-u*θ)-1))/(exp(-θ)-1)+1))
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (θ*(exp(-u*θ)-1)*(exp(-v*θ)-1)*exp(-(v*θ)-u*θ))/((exp(-θ)-1)^2*(((exp(-u*θ)-1)*(exp(-v*θ)-1))/(exp(-θ)-1)+1)^2)-(θ*exp(-u*θ)*exp(-v*θ))/((exp(-θ)-1)*(((exp(-u*θ)-1)*(exp(-v*θ)-1))/(exp(-θ)-1)+1))
        else
            valor = NaN
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int) # General method, see Nelsen (2006) p.40
        u = rand(n)
        t = rand(n)
        v = @. qinv∂uC(t,u)
        return hcat(u, v)
    end
    return (name = "Frank", param = θ, paramspace = Ω, taildep = tail, arch = archimedean,
            strict = strict, gen = ϕ, comp = comprehensive, cases = cases, pcases = paramcases,
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC 
    )
end


## Archimedan family 4.2.12 (Nelsen, 2006) ######

function copA4212(θ::AbstractFloat)
    # dependencies: `biseccion`
    if θ < 1.0
        error("Parameter must be ≥ 1.0")
        return nothing
    end
    Ω = ("θ ∈ [1, ∞)", 1.0, Inf)
    tail = true
    archimedean = true
    strict = "always"
    comprehensive = false
    cases = ["H", "M"]
    paramcases = [1.0, Inf]
    function ϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 ≤ t ≤ 1.0
            valor = (1/t - 1)^θ
        end
        return valor
    end
    function invϕ(z::AbstractFloat)
        valor = NaN
        if z ≥ 0.0
            valor = 1 / (1 + z^(1/θ))
        end
        return valor
    end
    function dϕ(t::AbstractFloat)
        valor = NaN
        if 0.0 < t < 1.0
            valor = -(((1/t-1)^(θ-1)*θ)/t^2)
        end
        return valor
    end
    function invdϕ(y::AbstractFloat) # approximation through bisection method
        valor = NaN
        if y < 0.0
            g(t) = dϕ(t) - y
            valor = biseccion(g, 0.0000000001, 0.999999999).raiz
        end 
        return valor 
    end
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = 1.0 / (1.0 + ((1.0/u - 1.0)^θ + (1.0/v - 1.0)^θ)^(1.0/θ))
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = ((1/u-1)^(θ-1)*((1/v-1)^θ+(1/u-1)^θ)^(1/θ-1))/(u^2*(((1/v-1)^θ+(1/u-1)^θ)^(1/θ)+1)^2)
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            w = invdϕ(dϕ(u) / t)
            valor = invϕ(ϕ(w) - ϕ(u))
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (((1/v-1)^θ+(1/u-1)^θ)^(1/θ-1)*(1/v-1)^(θ-1))/((((1/v-1)^θ+(1/u-1)^θ)^(1/θ)+1)^2*v^2)
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = (2*(1/u-1)^(θ-1)*((1/v-1)^θ+(1/u-1)^θ)^(2/θ-2)*(1/v-1)^(θ-1))/(u^2*(((1/v-1)^θ+(1/u-1)^θ)^(1/θ)+1)^3*v^2)-((1/u-1)^(θ-1)*((1/v-1)^θ+(1/u-1)^θ)^(1/θ-2)*(1/v-1)^(θ-1)*(1/θ-1)*θ)/(u^2*(((1/v-1)^θ+(1/u-1)^θ)^(1/θ)+1)^2*v^2)
        else
            valor = NaN
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int) # Genest & MacKay (1986) algorithm, see Nelsen (2006) p.134
        u = rand(n)
        t = rand(n)
        w = @. invdϕ(dϕ(u) / t)
        v = @. invϕ(ϕ(w) - ϕ(u))
        return hcat(u, v)
    end
    return (name = "A4212", param = θ, paramspace = Ω, taildep = tail, arch = archimedean,
            strict = strict, gen = ϕ, comp = comprehensive, cases = cases, pcases = paramcases,
            copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC, dcopula = ∂uvC, 
            diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC 
    )
end


## FLIPPING COPULAS ####################

function copFlip(K::NamedTuple, flip::String)
    # `K` a copula tuple generated with previous functions
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if flip == "h"
                valor = v - K.copula(1-u,v)
            elseif flip == "v"
                valor = u - K.copula(u,1-v)
            elseif flip == "hv"
                valor = u + v - 1 + K.copula(1-u,1-v)
            elseif flip == "s"
                valor = K.copula(v,u)
            end
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if flip == "h"
                valor = K.ducopula(1-u,v)
            elseif flip == "v"
                valor = 1 - K.ducopula(u,1-v)
            elseif flip == "hv"
                valor = 1 - K.ducopula(1-u,1-v)
            elseif flip == "s"
                valor = K.dvcopula(v,u)
            end
        else
            valor = NaN
        end
        return valor 
    end
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            g(v) = ∂uC(u,v) - t
            valor = biseccion(g, 0.000000001, 0.999999999).raiz
        else
            valor = NaN
        end
        return valor
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat) 
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if flip == "h"
                valor = 1- K.dvcopula(1-u,v)
            elseif flip == "v"
                valor = K.dvcopula(u,1-v)
            elseif flip == "hv"
                valor = 1 - K.dvcopula(1-u,1-v)
            elseif flip == "s"
                valor = K.ducopula(v,u)
            end
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if flip == "h"
                valor = K.dcopula(1-u,v)
            elseif flip == "v"
                valor = K.dcopula(u,1-v)
            elseif flip == "hv"
                valor = K.dcopula(1-u,1-v)
            elseif flip == "s"
                valor = K.dcopula(v,u)
            end
        else
            valor = NaN
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function rC(n::Int) # General method, see Nelsen (2006) p.40
        u = rand(n)
        t = rand(n)
        v = @. qinv∂uC(t,u)
        return hcat(u, v)
    end
    nombre = flip * "-flip(" * K.name * ")"
    return (name = nombre, copula = C, ducopula = ∂uC, invducopula = qinv∂uC, dvcopula = ∂vC,
            dcopula = ∂uvC, diag1 = δ1, diag2 = δ2, sver = ve, shor = ho, sim = rC, origcop = K 
    )
end


## CONVEX LINEAR COMBINATION OF TWO COPULAS ##############

function copConvex(β::AbstractFloat, C1::NamedTuple, C2::NamedTuple)
    # `C1` and `C2` copula tuples generated with previous functions
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = β*C1.copula(u,v) + (1-β)*C2.copula(u,v)
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = β*C1.ducopula(u,v) + (1-β)*C2.ducopula(u,v)
        else
            valor = NaN
        end
        return valor 
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = β*C1.dvcopula(u,v) + (1-β)*C2.dvcopula(u,v)
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            valor = β*C1.dcopula(u,v) + (1-β)*C2.dcopula(u,v)
        else
            valor = NaN
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            g(v) = ∂uC(u,v) - t
            valor = biseccion(g, 0.000000001, 0.999999999).raiz
        else
            valor = NaN
        end
        return valor
    end
    function rC(n::Int) 
        n1 = count(rand(n) .≤ β)
        n2 = n - n1
        s1 = C1.sim(n1)
        s2 = C2.sim(n2)
        return vcat(s1, s2) 
    end
    nombre = "convex(" * C1.name * "," * C2.name * ")"
    return (name = nombre, param = β, copula = C, ducopula = ∂uC, invducopula = qinv∂uC,
            dvcopula = ∂vC, dcopula = ∂uvC, diag1 = δ1, diag2 = δ2, sver = ve, shor = ho,
            sim = rC, copula1 = C1, copula2 = C2
    )
end


## VERTICAL GLUING OF TWO COPULAS #############

function copGluing(β::AbstractFloat, C1::NamedTuple, C2::NamedTuple)
    # `C1` and `C2` copula tuples generated with previous functions
    function C(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if u ≤ β
                valor = β*C1.copula(u/β,v)
            else
                valor = (1-β)*C2.copula((u-β)/(1-β),v) + β*v
            end
        else
            valor = NaN
        end
        return valor
    end
    function ∂uC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if u ≤ β
                valor = C1.ducopula(u/β,v)
            else
                valor = C2.ducopula((u-β)/(1-β),v)
            end
        else
            valor = NaN
        end
        return valor 
    end
    function ∂vC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if u ≤ β
                valor = β*C1.dvcopula(u/β,v)
            else
                valor = (1-β)*C2.dvcopula((u-β)/(1-β),v) + β
            end
        else
            valor = NaN
        end
        return valor 
    end
    function ∂uvC(u::AbstractFloat, v::AbstractFloat)
        if 0.0 ≤ u ≤ 1.0 && 0.0 ≤ v ≤ 1.0
            if u ≤ β
                valor = C1.dcopula(u/β,v)
            else
                valor = C2.dcopula((u-β)/(1-β),v)
            end
        else
            valor = NaN
        end
        return valor
    end
    δ1(t::AbstractFloat) = C(t, t)
    δ2(t::AbstractFloat) = C(t, 1.0 - t)
    ve(v::AbstractFloat, u::AbstractFloat = 0.5) = C(u, v)
    ho(u::AbstractFloat, v::AbstractFloat = 0.5) = C(u, v)
    function qinv∂uC(t,u)
        if (0.0 < u < 1.0) && (0.0 < t < 1.0)
            g(v) = ∂uC(u,v) - t
            valor = biseccion(g, 0.000000001, 0.999999999).raiz
        else
            valor = NaN
        end
        return valor
    end
    function rC(n::Int) # General method, see Nelsen (2006) p.40
        u = rand(n)
        t = rand(n)
        v = @. qinv∂uC(t,u)
        return hcat(u, v)
    end
    nombre = "gluing(" * C1.name * "," * C2.name * ")"
    return (name = nombre, param = β, copula = C, ducopula = ∂uC, invducopula = qinv∂uC,
            dvcopula = ∂vC, dcopula = ∂uvC, diag1 = δ1, diag2 = δ2, sver = ve, shor = ho,
            sim = rC, copula1 = C1, copula2 = C2
    )
end


; # end of file