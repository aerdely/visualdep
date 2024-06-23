### Bivariate empirical copula calculations
### Author: Arturo Erdely
### Version date: 2023-09-30

#=
   Auxiliary functions
   -------------------
   repetidos  resumen  desempate!  desempate  rangos  Fn

   Main functions
   --------------
   rangobs  copem  diagem  fcopem  condepcopem  condiagem
=#


using Statistics, LinearAlgebra


## Auxiliary functions

# repetidos (1 method)
"""
    repetidos(x::Vector)

Returns a type `Int` vector with the positions in vector `x` of repeated values.

## Examples
```
repetidos([1, 2, 3, 4, 1, 2, 5, 2])

repetidos(['A', 'B', 'C', 'A', 'B', 'A'])

repetidos(rand(10_000)) # no repeated values
```
"""
function repetidos(x::Vector)
    if allunique(x)
        return Int[]
    else
        u = unique(x)
        n = length(u)
        xpos = Int[]
        for k ∈ 1:n
            ix = findall(x .== u[k])
            if length(ix) > 1
                append!(xpos, ix[2:end])
            end
        end
        sort!(xpos)
    end
    return xpos 
end

# resumen (4 methods)
"""
    resumen(v::Vector{<:Real}, desplegar = false)

Calculates a tuple of summary statistics for the values in vector `v`.
If `desplegar = true` it also displays a summary table. The calculated
statistics are: sample size, total number of repeated values, mean,
minimum, first quartile, median, third quartile and maximum.

*Dependencies*:
- function `repetidos` 
- package `Statistics` (Julia standard library)

## Example
```
v = rand(rand(50), 55);
r = resumen(v, true)
r.median
```
"""
function resumen(v::Vector{<:Real}, desplegar = false)
    # dependencies: `repetidos` `Statistics`
    n = length(v)
    rep = length(repetidos(v))
    μ = mean(v)
    m = minimum(v)
    Q1 = quantile(v, 0.25)
    Q2 = median(v)
    Q3 = quantile(v, 0.75)
    M = maximum(v)
    r = (samplesize = n, repeated = rep, mean = μ, min = m,
         quartile_1 = Q1, median = Q2, quartile_3 = Q3, max = M)
    if desplegar
        display(hcat(collect(keys(r)), collect(values(r))))
    end
    return r
end

"""
    resumen(matriz::Matrix{<:Real}, desplegar = false)

Calculates a tuple of summary statistics for each column in `matriz`.
If `desplegar = true` it also displays a summary table. The calculated
statistics are: sample size, total number of repeated values, mean,
minimum, first quartile, median, third quartile and maximum.

*Dependencies*:
- function `resumen(v::Vector{<:Real},...)`
- package `Statistics` (Julia standard library)

## Example
```
M = [[.1,.3,.2,.1,.2,.4] [.6,.4,.2,.5,.3,.1] [.4,.5,.4,.6,.7,.8]]
r = resumen(M, true)
r.repeated
r.table
```
"""
function resumen(matriz::Matrix{<:Real}, desplegar = false)
    # dependencies: `Statistics` `resumen(v::Vector{<:Real},...)`
    nfilas, ncols = size(matriz)
    v = collect(1:ncols)
    n = fill(nfilas, ncols); rep = fill(0, ncols)
    μ = fill(0.0, ncols); m = fill(0.0, ncols)
    Q1 = fill(0.0, ncols); Q2 = fill(0.0, ncols)
    Q3 = fill(0.0, ncols); M = fill(0.0, ncols)
    D = Matrix{Any}(undef, nfilas, 0)
    for j ∈ 1:ncols
        r = resumen(matriz[:, j])
        n[j] = r.samplesize; rep[j] = r.repeated
        μ[j] = r.mean; m[j] = r.min
        Q1[j] = r.quartile_1; Q2[j] = r.median
        Q3[j] = r.quartile_3; M[j] = r.max
        if j == 1
            D = vcat([:variable j], hcat(collect(keys(r)), collect(values(r))))
        else
            D = hcat(D, pushfirst!(collect(values(r)), j))
        end
    end
    rr = (variable = v, samplesize = n, repeated = rep, mean = μ, min = m,
         quartile_1 = Q1, median = Q2, quartile_3 = Q3, max = M, table = D)
    if desplegar
        display(D)
    end
    return rr
end


# desempate! (4 methods)   desempate (4 methods)
"""
    desempate!(v::Vector{<:AbstractFloat}, ε = minimum(abs.(v)) / 1_000_000.0)

Adds (in place) a uniform random noise ±`ε` to the repeated values in `v` (if any). 
If `ε` is not specified then it defaults to `ε = minimum(abs.(v)) / 1_000_000`.

*Dependencies*: 
- function `repetidos`

## Example
```
a = [.1, .2, .1, .1, .3, .2]
desempate!(a)
println(a)
```
"""
function desempate!(v::Vector{<:AbstractFloat}, ε = minimum(abs.(v)) / 1_000_000.0)
    # dependencies: `repetidos`
    ipos = repetidos(v)
    npos = length(ipos)
    v[ipos] .+= ε .* (2 .* rand(npos) .- 1)
    return v
end

"""
    desempate!(matriz::Matrix{<:AbstractFloat}, ε = minimum(abs.(matriz)) / 1_000_000.0)

Adds (in place) a uniform random noise ±`ε` to the repeated values in each column of `matriz`
(if any). If `ε` is not specified then it defaults to `ε = minimum(abs.(matriz)) / 1_000_000`.

*Dependencies*: 
- function `repetidos`
- function `desempate!(v::Vector{<:AbstractFloat},...)`

## Example
```
A = [[.1,.2,.1,.3] [.1,.2,.3,.4] [.1,.1,.2,.1]]
println(A)
desempate!(A)
println(A)
```
"""
function desempate!(matriz::Matrix{<:AbstractFloat}, ε = minimum(abs.(matriz)) / 1_000_000.0)
    # dependencies: `repetidos` `desempate!(v::Vector{Float64},...)`
    ncol = size(matriz)[2]
    for j ∈ 1:ncol
        matriz[:, j] = desempate!(matriz[:, j], ε)
    end
    return(matriz)
end

"""
    desempate(v::Vector{<:AbstractFloat}, ε = minimum(abs.(v)) / 1_000_000.0)

Adds a uniform random noise ±`ε` to the repeated values in a copy of `v` (if any).
If `ε` is not specified then it defaults to `ε = minimum(abs.(v)) / 1_000_000`.
Returns a tuple with such copy but with no repeated values, a vector with the 
positions where repeated values were found, and the total number of repeated values.

*Dependencies*: 
- function `repetidos`

## Example
```
a = [.1, .2, .1, .1, .3, .2]
b = desempate(a)
a ≠ b[1] 
```
"""
function desempate(v::Vector{<:AbstractFloat}, ε = minimum(abs.(v)) / 1_000_000.0)
    # dependencies: `repetidos`
    vv = copy(v)
    ipos = repetidos(vv)
    npos = length(ipos)
    vv[ipos] .+= ε .* (2 .* rand(npos) .- 1)
    return (vv, ipos, npos)
end

"""
    desempate(matriz::Matrix{<:AbstractFloat}, ε = minimum(abs.(matriz)) / 1_000_000.0)

Adds a uniform random noise ±`ε` to the repeated values in each column of
a copy of `matriz` (if any). If `ε` is not specified then it defaults to
`ε = minimum(abs.(v)) / 1_000_000`. Returns a tuple with such copy but with no repeated values,
a vector with the positions where repeated values were found in each column,
and the total number of repeated values in each column.

*Dependencies*: 
- function `repetidos`
- function `desempate(v::Vector{<:AbstractFloat},...)`
        
## Example
```
A = [[.1,.2,.1,.3] [.1,.2,.3,.4] [.1,.1,.2,.1]]
R = desempate(A)
R[1]
A .≠ R[1]
R[2]
R[3]
```
"""
function desempate(matriz::Matrix{<:AbstractFloat}, ε = minimum(abs.(matriz)) / 1_000_000.0)
    # dependencies: `repetidos` `desempate(v::Vector{<:AbstractFloat},...)`
    d = size(matriz)
    M = zeros(d)
    ipos = Vector{Vector{Int}}(undef, d[2])
    npos = Vector{Int}(undef, d[2])
    for j ∈ 1:d[2]
        des = desempate(matriz[:, j], ε)
        M[:, j] = des[1]
        ipos[j] = des[2]
        npos[j] = des[3]
    end
    return (M, ipos, npos)
end


# rangos (1 method)
"""
    rangos(v::Vector{Any})

Returns a vector of positive integers with the ranks of the values in `v`
(in case it is possible to order them).

## Examples
```
v = [0.5, 0.1, 0.4, 0.2, 0.3]; rangos(v)
v = ['C', 'B', 'A', 'D', 'C']; rangos(v) 
v = [x -> x^2+1, "Hola", 4.3]; rangos(v) # not possible to order
```
"""
function rangos(v::AbstractVector)
    n = length(v)
    r = zeros(Int64, n)
    try
        iord = sortperm(v)
        D = Dict(iord .=> 1:n)
        for k ∈ 1:n
            r[k] = D[k]
        end
    catch
        @error "Vector elements cannot be ordered"
    finally 
        return r
    end
end

# Fn (empirical cumulative distribution function a.k.a ecdf) (2 methods)
"""
    Fn(x::Real, xobs::Vector{<:Real})

Empirical cumulative distribution function (ecdf) evaluated in a 
single value `x` based on an observed sample `xobs`. 

# Example
```
Fn(0, randn(10_000)) # approximately 0.5 = P(Z ≤ 0) where Z ~ Normal(0,1) 
```
"""
function Fn(x::Real, xobs::Vector{<:Real})
    count(xobs .≤ x) / length(xobs)
end

"""
    Fn(xvec::Vector{<:Real}, xobs::Vector{<:Real})

Empirical cumulative distribution function (ecdf) evaluated in values of `xvec`
based on an observed sample `xobs`. Returns a 2-tuple: a vector with counts
of how many observed values are less or equal to each value in `xvec`, and the
sample size, instead of calculating the ratios in order to keep all calculations
with integer type, for more efficiency and accuracy.

## Example
```
xobs = randn(10_000);
v = Fn([-2, 2], xobs)
(v[1][2] - v[1][1]) / v[2] # approximately 0.954 = P(-2 ≤ Z ≤ 2) where Z ~ Normal(0,1)
```
"""
function Fn(xvec::Vector{<:Real}, xobs::Vector{<:Real})
    m = length(xvec)
    Fx = zeros(Int, m)
    for k ∈ 1:m
        Fx[k] = count(xobs .≤ xvec[k])
    end
    return (Fx, length(xobs))
end 


## Main functions

# rangobs (1 method)
"""
    rangobs(matobs::Matrix{<:Real})

Assuming that each row of `matobs` is a sample observation of a continuous random vector of
dimension equal to the number of columns, this function returns a matrix of the same
dimensions but replacing the observations of each variable (column) by their marginal ranks. 
There should be no repeated values in the columns of `matobs` so check that first using the
`desempate` function and in case there are repeated values apply `desempate!` first.

This matrix divided by the sample size (number of rows) is equal to the pseudo-observations
of the underlying copula. But it is better to keep the counts without dividing by the sample
size in order to keep the data as integer type, for the sake of efficiency and accuracy
in the calculations.

*Dependencies*: 
- function `rangos`

## Example
```
X = rand(10);
ε = 0.2 .* randn(10);
Y = X .+ ε;
Z = randn(10);
[X Y Z]
P = rangobs([X Y Z])
```
"""
function rangobs(matobs::Matrix{<:Real})
    # dependencies: `rangos`
    n, d = size(matobs)
    P = zeros(Int, n, d)
    for j ∈ 1:d
        P[:, j] = rangos(matobs[:, j])
    end
    return P
end

# copem (1 method)
"""
    copem(matXY::Matrix{<:Real})

Generates a square matrix with bivariate empirical copula counts from a random sample of a 
bivariate continuous random vector (X,Y) where its observed values are the rows of `matXY`.
Returns the values of n⋅C(i/n, j/n) for i,j ∈ {1,...,n} where n is the sample size.
An important assumption is that (X,Y) are continuous random variables and there should be
no repeated values in the sample, that is, no repeated values per column in `matXY`. Check for that with the function `desempate` and if 
necessary apply `desempate!` first to the columns of `matXY`.

*Dependencies*: 
- function `rangobs`

## Example
```
X = randn(10);
ε = 0.2 .* randn(10);
Y = X .+ ε;
matXY = [X Y]
copem(matXY)
```
"""
function copem(matXY::Matrix{<:Real})
    # dependencies: `rangobs`
    n = size(matXY)[1] # sample size
    C = zeros(Int, n, n)
    R = rangobs(matXY[:, 1:2])
    for k ∈ 1:n
        C[R[k,1], R[k,2]] += 1
    end
    C = accumulate(+, C, dims = 1)
    C = accumulate(+, C, dims = 2)
    return C
end

# diagem (1 method)
"""
    diagem(C::Matrix{<:Integer})

Calculates the empirical diagonal section counts from a given empirical count copula `C`
generated by the function `copem`. Returns a 3-column matrix: 
- Column 1: values from 0 to n, where n is the sample size
- Column 2: values of n⋅C(i/n,i/n) for i ∈ {0,...,n}
- Column 3: values of n⋅C(i/n, 1-i/n) for i ∈ {0,...,n}

## Example
```
X = rand(10);
Y = @. -X + 0.2*rand();
matXY = [X Y];
C = copem(matXY)
diagem(C)
```
"""
function diagem(C::Matrix{<:Integer})
    # C = empirical copula counts generated by `copem`
    n = size(C)[1] # sample size
    D = zeros(Int, n+1, 3)
    D[:, 1] = collect(0:n)
    D[2:(n+1), 2] = diag(C)
    D[2:n, 3] = diag(C[:, n:-1:1], 1)
    return D 
end

# fcopem (1 method)
"""

    fcopem(i, j, matR::Matrix{<:Integer})

Calculates the bivariate empirical copula counts from a given `matR` of bivariate observed ranks.
Given (i,j) ∈ {1,...,n}^2 returns the value of n⋅C(i/n,j/n) where n is the sample size. You should
previously calculate `matR = rangobs(matXY)` where `matXY` is the original observed bivariate sample.

## Example
```
X = randn(10);
ε = 0.3 .* randn(10);
Y = X .+ ε;
matXY = [X Y]
matR = rangobs(matXY)
fcopem(5, 3, matR)
fcopem(0, 4, matR), fcopem(10, 4, matR)
```
"""
function fcopem(i, j, matR::Matrix{<:Integer})
    # matR = rangobs(matXY) 
    # matrix of sample ranks from (X,Y)
    n = size(matR)[1] # sample size
    c = 0
    for k ∈ 1:n
        c += matR[k,1] ≤ i && matR[k,2] ≤ j
    end
    return c
end

# condepcopem
"""

    condepcopem(C::Matrix{<:Integer})

Calculates empirical estimates of Spearman's *concordance* measure and Schweizer-Wolff's
*dependence* measure from an empirical copula matrix `C` calculated by the function `copem`.
Returns a named tuple `(con, dep)`.

## Example
```
X = rand(100);
Y = @. X*(1-X) + 0.5*rand();
matXY = [X Y];
C = copem(matXY);
condepcopem(C)
```
"""
function condepcopem(C::Matrix{<:Integer})
    # C = empirical copula counts generated by `copem`
    n = size(C)[1] # sample size
    v = collect(1:n)
    Π = v * transpose(v)
    wn = 12 / ((n^2)*(n^2-1))
    ρ = sum(n*C - Π) * wn # Spearman
    σ = sum(abs.(n*C - Π)) * wn # Schweizer
    return (con = ρ, dep = σ)
end

# condiagem
"""

    condiagem(D::Matrix{<:Integer})

Calculates an empirical estimate of Gini's *concordance* measure from the diagonals of
the empirical copula given in a matrix `D` calculated by the function `diagem`.
Returns a named tuple `(g1, g2, gini)` where `gini = g1 - g2` and where `g1` and `g2`
are the discordances from the Fréchet-Hoeffding lower and upper bounds, respectively,
for bivariate copula fuctions.
    
## Example
```
X = rand(100);
Y = @. X*(1-X) + 0.5*rand();
matXY = [X Y];
C = copem(matXY);
D = diagem(C);
condiagem(D)
```
""" 
function condiagem(D::Matrix{<:Integer})
    # D = empirical diagonal copula counts generated by `diagem`
    n = size(D)[1] - 1 # sample size
    wn = floor(n^2 / 2)
    γ1 = 2 * sum(D[:, 3]) / wn 
    γ2 = 2 * sum(D[:, 1] - D[:, 2]) / wn
    γ = γ1 - γ2
    return(g1 = γ1, g2 = γ2, gini = γ)
end

; # end of file