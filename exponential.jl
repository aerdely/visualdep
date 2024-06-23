## Exponential distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvExponential

"""
    rvExponential(μ::Real, σ::Real)

Location-dispersion **Exponential** probability distribution
with location parameter `μ` and dispersion parameter `σ > 0`, 
and with probability density function given by:

```math
f(x) = (1/σ) * exp(-(x - μ) / σ) * 1(x ≥ μ)
```

For example, if we define `X = rvExponential(3, 2)` then:

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
X = rvExponential(3, 2);
keys(X)
println(X.model)
X.param
X.param.loc
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
function rvExponential(μ::Real, σ::Real)
    if σ ≤ 0
        error("Dispersion parameter σ must be positive")
        return nothing
    else
        fdp(x) = (x ≥ μ) * exp(-(x-μ)/σ) / σ
        fda(x) = (x ≥ μ) * (1 - exp(-(x-μ)/σ))
        ctl(u) = 0 ≤ u ≤ 1 ? μ - σ*log(1-u) : NaN
        alea(n::Integer) = ctl.(rand(n))
    end
    soporte = ("[$μ , ∞[", μ, Inf)
    mediana = ctl(0.5)
    ric = ctl(0.75) - ctl(0.25)
    moda = μ
    media = μ + σ
    varianza = σ^2
    return (model = "Exponential", param = (loc = μ, dis = σ), range = soporte,
            pdf = fdp, cdf = fda, qtl = ctl, sim = alea, median = mediana,
            iqr = ric, mode = moda, mean = media, var = varianza
    )
end

; # end of file