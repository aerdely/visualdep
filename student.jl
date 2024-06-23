## Student distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvStudent, rvStudent3, rvStudentMix

using Random, Distributions


"""
    rvStudent(ν::Real)

**Student** standard probability distribution (https://en.wikipedia.org/wiki/Student%27s_t-distribution)
with parameter `ν > 0` degrees of freedom (df). 

For example, if we define `X = rvStudent(3.5)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode
- `X.mean` = theoretical mean
- `X.var` = theoretical variance

Dependencies: 
> `Distributions` (external) package

## Example
```
X = rvStudent(3.5);
keys(X)
println(X.model)
X.param
X.param.df
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
X.mean, X.var
xsim = X.sim(10_000); # a random sample of size 10,000
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
using Statistics # Julia standard package
median(xsim), mean(xsim), var(xsim) # sample median, mean, and variance
```
"""
function rvStudent(ν::Real)
    if ν ≤ 0
        error("Parameter must be positive")
        return nothing
    else
        S = TDist(ν)
        dst(x) = pdf(S, x)
        pst(x) = cdf(S, x)
        qst(u) = 0 < u ≤ 1 ? quantile(S, u) : NaN
        rst(n::Integer) = qst.(rand(n))
    end
    soporte = ("]-∞ , ∞[", -Inf, Inf)
    mediana = 0.0
    ric = qst(0.75) - qst(0.25)
    moda = 0.0
    media = ν > 1 ? 0.0 : NaN
    if ν > 2
        varianza = ν / (ν-2)
    elseif 1 < ν ≤ 2
        varianza = Inf
    else
        varianza = NaN 
    end
    return (model = "Student", param = (df = ν,), range = soporte,
            pdf = dst, cdf = pst, qtl = qst, sim = rst,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end


"""
    rvStudent3(μ::Real, σ::Real, ν::Real)

Location-dispersion **Student** probability distribution with location
parameter `μ`, dispersion parameter `σ > 0`, and `ν > 0` degrees of freedom (df). 

For example, if we define `X = rvStudent3(-2.0, 1.7, 3.5)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode
- `X.mean` = theoretical mean
- `X.var` = theoretical variance

Dependencies: 
> `rvStudent` function

## Example
```
X = rvStudent3(-2.0, 1.7, 3.5);
keys(X)
println(X.model)
X.param
X.param.dis
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
X.mean, X.var
xsim = X.sim(10_000); # a random sample of size 10,000
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
using Statistics # Julia standard package
median(xsim), mean(xsim), var(xsim) # sample median, mean, and variance
```
"""
function rvStudent3(μ::Real, σ::Real, ν::Real)
    if min(σ, ν) ≤ 0
        error("Parameters σ and ν must be positive")
        return nothing
    else
        S = rvStudent(ν)
        dst3(x) = S.pdf((x - μ) / σ) / σ
        pst3(x) = S.cdf((x - μ) / σ)
        qst3(u) = 0 < u ≤ 1 ? μ +σ*S.qtl(u) : NaN
        rst3(n::Integer) = qst3.(rand(n))
    end
    soporte = ("]-∞ , ∞[", -Inf, Inf)
    mediana = μ
    ric = qst3(0.75) - qst3(0.25)
    moda = μ
    media = ν > 1 ? μ : NaN
    if ν > 2
        varianza = (σ^2)*ν / (ν-2)
    elseif 1 < ν ≤ 2
        varianza = Inf
    else
        varianza = NaN 
    end
    return (model = "Student3", param = (loc = μ, dis = σ, df = ν), range = soporte,
            pdf = dst3, cdf = pst3, qtl = qst3, sim = rst3,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end


"""
    biseccion(f::Function, a::Real, b::Real; δ::Real = abs((a + b)/2) / 1_000_000, m::Int = 10_000)

`f` a continuous function over a closed interval `[a,b]` such that `f(a)f(b)<0`
with a maximum error tolerance `δ` to a value `c` such that `f(c)=0` in a 
maximum number of `m` iterations (bisection method). 

> Warning: If `c` is not unique this algorithm only finds one of them.

## Example
```
f(x) = (x - 3) * (x - 1) * (x + 1)
biseccion(f, -1.9, 0)
```
"""
function biseccion(f::Function, a::Real, b::Real; δ::Real = abs((a + b)/2) / 1_000_000, m::Int = 10_000)
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
        println("The maximum number of $m iterations has been reached")
    end
    return (raiz = z, dif = f(z), numiter = iter, maxiter = m, tol = δ)
end


"""

    rvStudentMix(w::Real, μ1::Real, σ1::Real, ν1::Real, μ2::Real, σ2::Real, ν2::Real)

Convex linear combination of two three-paramater Student3 distributions:

```math
    w*Student3(μ1,σ1,ν1) + (1-w)*Student3(μ2,σ2,ν2) ,      0 < w < 1
```

For example, if we define
`X = rvStudentMix(0.7, -2, 7, 20, 5, 1, 3)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mean` = theroetical mean (if it exists)
- `X.modes` = theoretical modes

Dependencies:
> function `rvStudent3`

> Julia standard package `Random`

## Example
```
X = rvStudentMix(0.7, -2, 7, 20, 5, 1, 3);
keys(X)
println(X.model)
X.param
X.param.dis2
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking the interquartile range
X.modes # two modes in this case
X.mean
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
sum(xsim) / 10_000 # sample mean
```
"""
function rvStudentMix(w::Real, μ1::Real, σ1::Real, ν1::Real, μ2::Real, σ2::Real, ν2::Real)
    # Dependencies: functions `rvStudent3` and `biseccion`
    if min(σ1, ν1) ≤ 0 || min(σ2, ν2) ≤ 0
        error("Parameters `σ` and `ν` must be positive in both cases")
        return nothing
    elseif w ≤ 0 || w ≥ 1
        error("Weight must be `0 < w < 1`")
        return nothing
    else
        X = rvStudent3(μ1, σ1, ν1)
        Y = rvStudent3(μ2, σ2, ν2)
        d(x) = w*X.pdf(x) + (1-w)*Y.pdf(x)
        p(x) = w*X.cdf(x) + (1-w)*Y.cdf(x)
        function r(n::Integer)
            y = Y.sim(n)
            ix = randsubseq(1:n, w)
            x = X.sim(length(ix))
            y[ix] = x
            return y
        end
        function q(u)
            q1, q2 = minmax(X.qtl(u), Y.qtl(u))
            g(x) = p(x) - u
            return biseccion(g, q1, q2).raiz
        end
        mediana = q(0.5)
        ric = q(0.75) - q(0.25)
        media = w*X.mean + (1-w)*Y.mean
        soporte = ("]-∞ , +∞[", -Inf, Inf)
        X.mode == Y.mode ? mod = (X.mod,) : mod = (X.mode, Y.mode)
        return (model = "StudentMix",
            param = (w = w, loc1 = μ1, dis1 = σ1, df1 = ν1, loc2 = μ2, dis2 = σ2, df2 = ν2),
            range = soporte, pdf = d, cdf = p, qtl = q, sim = r, median = mediana, iqr = ric,
            mean = media, modes = mod 
        )
    end
end


; # end of file