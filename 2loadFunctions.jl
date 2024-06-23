### Visual analysis of bivariate dependence between continuous random variables
### Authors: Arturo Erdely and Manuel Rubio-Sánchez
### Version date: 2024-06-23

### LOAD REQUIRED FUNCTIONS

## Auxiliary packages and files

begin
    using CSV, DataFrames, Distributions, StatsPlots
    include("copem2D.jl")
    using .copem2D
    include("copula2Dfams.jl")
    include("rv.jl")
end;


## Auxiliary functions and objects

function sklar(copula, marginal1, marginal2)
    simC(n::Int) = copula.sim(n)
    function simXY(n::Int)
        UV = simC(n)
        X = marginal1.qtl.(UV[:, 1])
        Y = marginal2.qtl.(UV[:, 2])
        return hcat(X, Y)
    end
    regC(a, u) = copula.invducopula(a, u)
    regXY(a, x) = marginal2.qtl(regC(a, marginal1.cdf(x)))
    return (simC = simC, simXY = simXY, regC = regC, regXY = regXY)
end;

function sklarplot(copula, marginal1, marginal2; nsim = 1_000, summary = true)
    s = sklar(copula, marginal1, marginal2)
    XY = s.simXY(nsim)
    x = XY[:, 1];
    y = XY[:, 2];
    d = dstat(x, y);
    if summary
        display(d.summary)
    end
    dp = dplot(d)
    plot(dp.all)
end;

# univariate marginal distributions
begin
    M0 = rvUniform(0, 1) # uniform (0,1)
    M1 = rvExponential(1, 2) # monotone, non-heavy tail
    M2 = rvPareto(2, 3) # monotone, heavy tail
    M3 = rvStudent3(3, 1.5, 50) # non-monotone, non-heavy tail
    M4 = rvStudent3(3, 1.5, 2.5) # non-monotone, heavy tail
    M5 = rvStudentMix(0.5, -16, 7, 20, 15, 1, 3) # bimodal bell-shaped
    M6 = rvKuma(0.25,0.15) # bimodal non-bell shaped
end;

function copulasurface(C, texto::String)
    surface(u, v, C, showaxis = false, grid = false, camera = (-27, 20), colormap = :RdBu, colorbar = false, legend = false)
    plot!([0, 0], [1, 1], [0, 1], color = :black, lw = 0.5)
    plot!([0, 1], [1, 1], [1, 1], color = :black, lw = 0.5)
    plot!([1, 1], [0, 0], [0, 1], color = :black, lw = 0.5)
    plot!([1, 1], [0, 1], [1, 1], color = :black, lw = 0.5)
    annotate!(0, 0, -0.02, text(0, 6, :top))
    annotate!(0, 1, -0.02, text(1, 6, :top))
    annotate!(1, 0, -0.02, text(1, 6, :top))
    annotate!(-0.05, 1, 1.05, text(1, 6, :top))
    annotate!(0.5, -0.03, 0, text("u", 6, :top))
    annotate!(-0.03, 0.5, 0, text("v", 6, :top))
    annotate!(0.5, 1, 1.2, text(texto, 8, :right))
end;

println("Functions loaded.", "\n")