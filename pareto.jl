## Pareto distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvPareto

"""
    rvPareto(a::Real, b::Real)

**Pareto** probability distribution with parameters `a > 0` and `b > 0`, 
and with probability density function given by:

```math
f(x) = (b * a^b) / (x^(b+1)) * 1(x ≥ a)
```

For example, if we define `X = rvPareto(2, 9)` then:

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

## Example
```
X = rvPareto(2, 9);
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
X.mean, X.var
xsim = X.sim(10_000); # a random sample of size 10,000
diff(sort(xsim)[[2499, 7500]])[1] # sample interquartile range
using Statistics # Julia standard package
median(xsim), mean(xsim), var(xsim) # sample median, mean, and variance
```
"""
function rvPareto(a::Real, b::Real)
    if min(a, b) ≤ 0
        error("Both parameters must be positive")
        return nothing
    else
        dpareto(x) = (x ≥ a) * (b * a^b) / (x^(b+1))
        ppareto(x) = (x ≥ a) * (1 - (a / x)^b)
        qpareto(u) = 0 < u ≤ 1 ? a / (1-u)^(1/b) : NaN
        rpareto(n::Integer) = qpareto.(rand(n))
    end
    soporte = ("[$a , ∞[", a, Inf)
    mediana = qpareto(0.5)
    ric = qpareto(0.75) - qpareto(0.25)
    moda = a
    media = b > 1 ? b*a / (b - 1) : Inf
    varianza = b > 2 ? (b * a^2) / ((b - 2) * (b - 1)^2) : Inf
    return (model = "Pareto", param = (a = a, b = b), range = soporte,
            pdf = dpareto, cdf = ppareto, qtl = qpareto, sim = rpareto,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end

; # end of file