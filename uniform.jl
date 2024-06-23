## Continuous uniform distribution in Julia
## Author: Arturo Erdely
## Version date: 2024-02-11

## Functions: rvUniform

"""
    rvUniform(a::Real, b::Real)

Continuous **Uniform** probability distribution over the closed
interval `[a,b]` with parameters `a < b`. 

For example, if we define `X = rvUniform(-1, 2)` then:

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
X = rvUniform(-1, 2);
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
function rvUniform(a::Real, b::Real)
    if a ≥ b
        error("Lower end of interval must be stricly smaller than the upper end")
    else
        dunif(x) = (a ≤ x ≤ b) / (b-a)
        punif(x) = (x-a)/(b-a)*(a ≤ x ≤ b) + 1*(x > b)
        qunif(u) = 0 < u ≤ 1 ? a + (b-a)*u : NaN
        runif(n::Integer) = a .+ (b-a) .* rand(n)
    end
    soporte = ("[$a , $b]", a, b)
    mediana = (a+b)/2
    ric = qunif(0.75) - qunif(0.25)
    moda = NaN
    media = (a+b)/2
    varianza = ((b-a)^2) / 12
    return (model = "Uniform", param = (a = a, b = b), range = soporte,
            pdf = dunif, cdf = punif, qtl = qunif, sim = runif,
            median = mediana, iqr = ric, mode = moda, mean = media, var = varianza
    )
end

; # end of file