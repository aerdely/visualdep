## Kumaraswamy distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvKuma, rvKuma4, rvKumaV, rvKumaV4, biseccion, rvKumaMix


using Random

"""
    rvKuma(a::Real, b::Real)

**Kumaraswamy** probability distribution with parameters `a > 0` 
and `b > 0`, and with probability density function given by:

```math
f(x) = a * b * (x^(a - 1)) * (1 - x^a)^(b - 1) * 1(0 < x < 1)
``` 

For example, if we define `X = rvKuma(2, 9)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode (returns `NaN` if it does not exist)

## Example
```
X = rvKuma(2, 9);
keys(X)
println(X.model)
X.param
X.param.a
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
```
"""
function rvKuma(a::Real, b::Real)
    if min(a, b) ≤ 0.0
        error("Both parameters must be positive")
        return nothing
    else
        dkuma(x) = a * b * (x^(a - 1)) * (1 - x^a)^(b - 1) * (0 < x < 1)
        pkuma(x) = (1 - (1 - x^a)^b)*(0 < x < 1) + 1*(x ≥ 1)
        qkuma(u) = (1 - (1 - u)^(1 / b))^(1 / a)
        rkuma(n) = qkuma.(rand(n))
        mediana = qkuma(0.5)
        if a ≥ 1 && b ≥ 1 && (a,b) ≠ (1,1)
            moda = ((a - 1) / (a*b - 1))^(1 / a)
        else
            moda = NaN
        end
        ric = qkuma(0.75) - qkuma(0.25) # interquartile range
        soporte = ("[0 , 1]", 0.0, 1.0)
        return (model = "Kuma", param = (a = a, b = b), range = soporte, 
            pdf = dkuma, cdf = pkuma, qtl = qkuma, sim = rkuma, 
            median = mediana, iqr = ric, mode = moda
        )
    end
end


"""
    rvKuma4(a::Real, b::Real, c::Real, d::Real)

4-parameter **Kumaraswamy** probability distribution
over a closed interval `[c,d]` with parameters `a > 0` and `b > 0` 
obtained a follows: If `K = rvKuma(a,b)` then:

```math
X = c + (d-c)*K
```

For example, if we define `X = rvKuma4(2, 9, -1, 4)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode (returns `NaN` if it does not exist)

Dependencies:
> function `rvKuma`

## Example
```
X = rvKuma4(2, 9, -1, 4);
keys(X)
println(X.model)
X.param
X.param.c
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
```
"""
function rvKuma4(a::Real, b::Real, c::Real, d::Real)
    # Dependencies: function `rvKuma`
    if min(a, b) ≤ 0
        error("Parameters `a` and `b` must be positive")
    elseif c ≥ d
        error("Left end of interval `c` must be strictly less than the right end `d`")
    else
        X = rvKuma(a, b)
        dkuma(x) = X.pdf((x-c)/(d-c)) / (d-c)
        pkuma(x) = X.cdf((x-c)/(d-c))
        qkuma(u) = c + (d-c)*X.qtl(u)
        rkuma(n) = qkuma.(rand(n))
        mediana = qkuma(0.5)
        if a ≥ 1 && b ≥ 1 && (a,b) ≠ (1,1)
            moda = c + (d-c)*X.mode
        else
            moda = NaN
        end
        ric = qkuma(0.75) - qkuma(0.25) # interquartile range 
        soporte = ("[$c , $d]", c, d)
        return (model = "Kuma4",
            param = (a = a, b = b, c = c, d = d), range = soporte, 
            pdf = dkuma, cdf = pkuma, qtl = qkuma, sim = rkuma,
            median = mediana, iqr = ric, mode = moda
        )
    end
end


"""
    rvKumaV(a::Real, b::Real)

Probability distribution of `1-K` where `K` is **Kumaraswamy**
with parameters `a > 0` and `b > 0`.

For example, if we define `X = rvKumaV(2, 9)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode (returns `NaN` if it does not exist)

## Example
```
X = rvKumaV(2, 9);
keys(X)
println(X.model)
X.param
X.param.b
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range 
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
```
"""
function rvKumaV(a::Real, b::Real)
    if min(a, b) ≤ 0
        error("Both parameters must be positive")
    else
        dk(x) = a * b * (x^(a - 1)) * (1 - x^a)^(b - 1) * (0 < x < 1)
        dkuma(x) = dk(1-x)
        pk(x) = (1 - (1 - x^a)^b)*(0 < x < 1) + 1*(x ≥ 1)
        pkuma(x) = 1 - pk(1-x)
        qk(u) = (1 - (1 - u)^(1 / b))^(1 / a)
        qkuma(u) = 1 - qk(1-u)
        rkuma(n) = qkuma.(rand(n))
        mediana = qkuma(0.5)
        if a ≥ 1 && b ≥ 1 && (a,b) ≠ (1,1)
            moda = 1 - ((a - 1) / (a*b - 1))^(1 / a)
        else
            moda = NaN
        end
        ric = qkuma(0.75) - qkuma(0.25) # ric = rango intercuartílico
        soporte = ("[0 , 1]", 0.0, 1.0)
        return (model = "KumaV", param = (a = a, b = b), range = soporte, 
            pdf = dkuma, cdf = pkuma, qtl = qkuma, sim = rkuma,
            median = mediana, iqr = ric, mode = moda
        )
    end
end


"""
    rvKumaV4(a::Real, b::Real, c::Real, d::Real)

Probability distribution of `1-K` where `K` is **Kumaraswamy**
with parameters `a > 0` and `b > 0`, over a closed interval `[c,d]`.
If `K = rvKumaV(a,b)` then:

```math
X = c + (d-c)*K
```

For example, if we define `X = rvKumaV4(2, 9, -1, 4)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.mode` = theoretical mode (returns `NaN` if it does not exist)

Dependencies:
> function `rvKumaV`

## Example
```
X = rvKumaV4(2, 9, -1, 4);
keys(X)
println(X.model)
X.param
X.param.d
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking interquartile range 
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range 
```
"""
function rvKumaV4(a::Real, b::Real, c::Real, d::Real)
    # Dependencies: function `rvKumaV`
    if min(a, b) ≤ 0
        error("Parameters `a` and `b` must be positive")
    elseif c ≥ d
        error("Left end of interval `c` must be strictly less than the right end `d`")
    else
        X = rvKumaV(a, b)
        dkuma(x) = X.pdf((x-c)/(d-c)) / (d-c)
        pkuma(x) = X.cdf((x-c)/(d-c))
        qkuma(u) = c + (d-c)*X.qtl(u)
        rkuma(n) = qkuma.(rand(n))
        mediana = qkuma(0.5)
        if a ≥ 1 && b ≥ 1 && (a,b) ≠ (1,1)
            moda = c + (d-c)*X.mode
        else
            moda = NaN
        end
        ric = qkuma(0.75) - qkuma(0.25) # ric = rango intercuartílico
        soporte = ("[$c , $d]", c, d)
        return (model = "KumaV4",
            param = (a = a, b = b, c = c, d = d), range = soporte, 
            pdf = dkuma, cdf = pkuma, qtl = qkuma, sim = rkuma,
            median = mediana, iqr = ric, mode = moda
        )
    end
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

    rvKumaMix(w::Real, a1::Real, b1::Real, c1::Real, d1::Real, a2::Real, b2::Real, c2::Real, d2::Real)

Convex linear combination of two four-paramater Kumaraswamy distributions:

```math
    w*rvKuma4(a1,b1,c1,d1) + (1-w)*rvKumaV4(a2,b2,c2,d2) ,      0 < w < 1
```

For example, if we define 
`X = rvKumaMix(0.3, 2, 9, -4, -1, 5, 5, 1, 3)` then:

- `X.model` = Name of probability distribution
- `X.param` = tuple of parameters
- `X.pdf(x)` = probability density function evaluated in `x`
- `X.cdf(x)` = cummulative distributionfunction evaluated in `x`
- `X.qtl(u)` = quantile function evaluated in value `0 ≤ u ≤ 1`
- `X.sim(n)` = vector of a size `n` random sample
- `X.range` = theoretical range (support)
- `X.median` = theoretical median
- `X.iqr` = interquartile range 
- `X.modes` = theoretical modes (returns `NaN` if it does not exist)

Dependencies:
> functions `rvKuma4`, `rvKumaV4` and `biseccion`

> Julia standard package `Random`

## Example
```
X = rvKumaMix(0.3, 2, 9, -4, -1, 5, 5, 1, 3);
keys(X)
println(X.model)
X.param
X.param.c2
println(X.range)
X.median
X.cdf(X.median) # checking the median
X.qtl(0.5) # checking a quantile
X.iqr
X.qtl(0.75) - X.qtl(0.25) # checking the interquartile range
xsim = X.sim(10_000); # a random sample of size 10,000
sum(sort(xsim)[5_000:5_001]) / 2 # sample median
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
X.modes # two modes in this case
```
"""
function rvKumaMix(w::Real, a1::Real, b1::Real, c1::Real, d1::Real, a2::Real, b2::Real, c2::Real, d2::Real)
    # Dependencies: functions `rvKuma4`, `rvKumaF4` and `biseccion`
    if min(a1, b1) ≤ 0 || min(a2, b2) ≤ 0
        error("Parameters `a` and `b` must be positive in both cases")
    elseif c1 ≥ d1 || c2 ≥ d2
        error("Left ends of intervals must be strictly less than right ends")
    elseif w ≤ 0 || w ≥ 1
        error("Weight must be `0 < w < 1`")
    else
        X = rvKuma4(a1,b1,c1,d1)
        Y = rvKumaV4(a2,b2,c2,d2)
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
        m1, m2 = min(c1,c2), max(d1,d2)
        soporte = ("[$m1 , $m2]", m1, m2)
        mod = (X.mode, Y.mode)
        return (model = "KumaMix",
            param = (w = w, a1 = a1, b1 = b1, c1 = c1, d1 = d1, a2 = a2, b2 = b2, c2 = c2, d2 = d2),
            range = soporte, pdf = d, cdf = p, qtl = q, sim = r, median = mediana, iqr = ric, modes = mod)
    end
end

; # end of file