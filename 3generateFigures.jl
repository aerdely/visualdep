### Visual analysis of bivariate dependence between continuous random variables
### Authors: Arturo Erdely and Manuel Rubio-Sánchez
### Version date: 2024-06-23

### GENERATE FIGURES

println("*** Generating 11 figures as .pdf files...", "\n")

## 3. Effect of marginal distributions on scatter plots

# Figure 1
begin
    n = 1_000
    x = zeros(n, 5)
    y = zeros(n, 5)
    u = rand(n)
    v = rand(n)
    x[:, 1] = u
    y[:, 1] = v
    x[:, 2] = M1.qtl.(u)
    y[:, 2] = M1.qtl.(v)
    x[:, 3] = M3.qtl.(u)
    y[:, 3] = M3.qtl.(v)
    x[:, 4] = M5.qtl.(u)
    y[:, 4] = M5.qtl.(v)
    x[:, 5] = M6.qtl.(u)
    y[:, 5] = M6.qtl.(v)
    xp = fill(plot(), 5)
    yp = fill(plot(), 5)
    for k ∈ 1:5
        xp[k] = histogram(x[:, k], legend = false, color = :steelblue3, alpha = 0.7, lw = 0.2, bins = range(minimum(x[:, k]), maximum(x[:, k]), length = 20), showaxis = false, grid = false, normalize = false)
        yp[k] = histogram(y[:, k], legend = false, color = :palegreen3, alpha = 0.7, lw = 0.2, permute = (:x, :y), yaxis = :flip, bins = range(minimum(y[:, k]), maximum(y[:, k]), length = 20), showaxis = false, grid = false)
    end
    p = fill(plot(), 5, 5)
    for i ∈ 1:5, j ∈ 1:5
        p[i, j] = scatter(x[:, i], y[:, j], legend = false, ms = 0.5, mc = :black, showaxis = false, grid = false)
    end
    pnull = plot(showaxis = false, grid = false)
    fig01 = plot(pnull, xp[1], xp[2], xp[3], xp[4], xp[5],
            yp[1], p[1,1], p[2,1], p[3,1], p[4,1], p[5,1],
            yp[2], p[1,2], p[2,2], p[3,2], p[4,2], p[5,2],
            yp[3], p[1,3], p[2,3], p[3,3], p[4,3], p[5,3],
            yp[4], p[1,4], p[2,4], p[3,4], p[4,4], p[5,4],
            yp[5], p[1,5], p[2,5], p[3,5], p[4,5], p[5,5],         
            layout = (6,6), size = (550, 450), margin = (0.0, :mm)
    )
    savefig("figure01.pdf")
    println("figure01.pdf", "\n")
end


## 4. Bivariate copulas: Fréchet-Hoeffding bounds and independence

# Figure 2 (a, b, c)
begin
    n = 50
    u = range(0, 1, length = n)
    v = u
    CΠ = zeros(n, n)
    CM = zeros(n, n)
    CW = zeros(n, n)
    Π = copΠ()
    M = copM()
    W = copW()
    for i ∈ 1:n, j ∈ 1:n
        CΠ[i, j] = Π.copula(u[i], v[j])
        CM[i, j] = M.copula(u[i], v[j])
        CW[i, j] = W.copula(u[i], v[j])
    end
    fig2a = copulasurface(CΠ, "Π(u,v)")
    fig2b = copulasurface(CM, "M(u,v)")
    fig2c = copulasurface(CW, "W(u,v)")
    fig02A = plot(fig2a, fig2b, fig2c, layout = (1,3))
    savefig("figure02A.pdf")
    println("figure02A.pdf", "\n")
end

# Figure 2 (a', b', c')
begin
    uv = Π.sim(1000)
    fig2ap = scatter(uv[:, 1], uv[:, 2], legend = false, mc = :black, ms = 1.0, grid = false, xticks = [0,1], yticks = [0,1])
    annotate!(0.5, 0.8, text("Π", 16, :red))
    uv = M.sim(300)
    fig2bp = scatter(uv[:, 1], uv[:, 2], legend = false, mc = :black, ms = 1.0, grid = false, xticks = [0,1], yticks = [0,1], ylabel = "v")
    annotate!(0.5, -0.1, text("u", 10))
    annotate!(0.4, 0.8, text("M", 16, :red))
    uv = W.sim(300)
    fig2cp = scatter(uv[:, 1], uv[:, 2], legend = false, mc = :black, ms = 1.0, grid = false, xticks = [0,1], yticks = [0,1])
    annotate!(0.6, 0.8, text("W", 16, :red))
    fig02B = plot(fig2ap, fig2bp, fig2cp, layout = (1,3), size = (600, 200))
    savefig("figure02B.pdf")
    println("figure02B.pdf", "\n")
end


## 5.4 Rank plots and empirical estimation

# Figure 3
begin
    n = 1_000
    R1 = copΠ().sim(n)
    pR1 = scatter(R1[:, 1], R1[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R1", labelfontsize = 8)
    R2 = copClayton(5.8).sim(n)
    pR2 = scatter(R2[:, 1], R2[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R2", labelfontsize = 8)
    R3 = copFlip(copA4212(2.5), "h").sim(n)
    pR3 = scatter(R3[:, 1], R3[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R3", labelfontsize = 8)
    R41 = copClayton(8.0)
    R42 = copFrank(-20.0)
    R4 = copConvex(0.5, R41, R42).sim(n)
    pR4 = scatter(R4[:, 1], R4[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R4", labelfontsize = 8)
    R5 = copConvex(0.5, copFrank(60.0), copΠ()).sim(n)
    pR5 = scatter(R5[:, 1], R5[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R5", labelfontsize = 8)
    R6 = copConvex(0.5, copFrank(-60.0), copΠ()).sim(n)
    pR6 = scatter(R6[:, 1], R6[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R6", labelfontsize = 8)
    R7 = copGluing(0.5, copFrank(20.0), copFrank(-20.0)).sim(n)
    pR7 = scatter(R7[:, 1], R7[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R7", labelfontsize = 8)
    R81 = copFlip(copClayton(5.8), "h")
    R82 = copA4212(2.5)
    R8 = copGluing(0.5, R81, R82).sim(n)
    pR8 = scatter(R8[:, 1], R8[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R8", labelfontsize = 8)
    R91 = copΠ()
    R92 = copFlip(copFlip(copClayton(5.8), "hv"), "v")
    R9 = copGluing(0.5, R91, R92).sim(n)
    pR9 = scatter(R9[:, 1], R9[:, 2], legend = false, size = (500,500), ms = 1, mc = :black, xlim = (0,1), ylim = (0,1), showaxis = false, grid = false, xlabel = "R9", labelfontsize = 8)
    pp = fill(plot(), 3, 3)
    fig03 = plot(pR1, pR2, pR3, pR4, pR5, pR6, pR7, pR8, pR9, layout = (3,3), size = (465, 500), margin = (0.0, :mm))
    savefig("figure03.pdf")
    println("figure03.pdf", "\n")
end


## 5.5 Diagonal sections

# Figure 4
begin 
    # main diagonal
    t = collect(range(0.0, 1.0, length = 1_001));
    dW = @. (2t - 1) * (t > 1/2)
    dP = t .^ 2
    dM = t
    a = 1.75
    plot(t, dM, label = " M", legend = (0.3, 0.8), xtickfont = 5, ytickfont = 5, color = :blue, xlabel = "t", ylabel = "δ(t)", lw = a)
    plot!(t, dP, color = :green, label = " Π", lw = a)
    p1 = plot!(t, dW, color = :red, label = " W", lw = a)
    # secondary diagonal
    d2W = zeros(length(t))
    d2P = @. t*(1-t)
    d2M = @. t*(t ≤ 0.5) + (1-t)*(t > 0.5)
    plot(t, d2M, legend = false, xtickfont = 5, ytickfont = 5, color = :blue, xlabel = "t", ylabel = "λ(t)", lw = a)
    plot!(t, d2P, color = :green, lw = a)
    p2 = plot!(t, d2W, color = :red, lw = a)
    # generate figure
    fig04 = plot(p1, p2, layout = (1, 2), size = (450, 220))
    savefig("figure04.pdf")
    println("figure04.pdf", "\n")
end



## 6.1 Example 1: Non quadrant dependence

# Figure 5
begin
    C = copGluing(0.5, copFrank(-30.0), copFrank(30.0))
    M6 = rvKuma(0.25,0.15) # bimodal Kumaraswamy marginal distribution
    M4 = rvStudent3(3, 1.5, 2.5) # unimodal t-Student (non-monotone), heavy tail
    println("-------------------------------")
    println("Copula: ", C.name)
    println("X-marginal: ", M6.model)
    println("Y-marginal: ", M4.model)
    println("-------------------------------")
    fig05 = sklarplot(C, M6, M4)
    savefig("figure05.pdf")
    println("figure05.pdf")
    s = sklar(C, M6, M4)
    XY = s.simXY(1_000)
    x = XY[:, 1]
    y = XY[:, 2]
    d = dstat(x, y)
    display(d.summary)
    println("\n")
end


## 6.2 Example 2: Positive quadrant dependence with noise

# Figure 6
begin
    X = rvPareto(2.0, 10.0)
    Z = rvPareto(2.0, 10.0)
    B = Bernoulli(0.4)
    n = 2_000
    xx = X.sim(n)
    zz = Z.sim(n)
    bb = rand(B, n)
    σ = 0.03
    ε = rand(Normal(0, σ), n)
    yy = @. (1-bb)*(xx + ε) + bb*zz
    d = dstat(xx, yy)
    p = dplot(d)
    fig06 = plot(p.all)
    savefig("figure06.pdf")
    println("figure06.pdf")
    display(d.summary)
    println("\n")
end



## 7.1 Example 3: NQD with outliers

# Figure 7
begin
    archivo = "example3dataset.csv"
    data = DataFrame(CSV.File(archivo))
    function transf(texto)
        parse(Float64, replace(texto, "\$" => "", "," => ""))
    end
    data.mwp = zeros(nrow(data))
    data.ceo = zeros(nrow(data))
    data.pr = zeros(nrow(data))
    for i ∈ 1:nrow(data)
        data.mwp[i] = transf(data.median_worker_pay[i]) / 1_000
        data.ceo[i] = transf(data.salary[i]) / 1_000
        data.pr[i] = (data.ceo[i] / data.mwp[i]) / 1_000
    end
    id_mwp = findall(data.mwp .> 0.0)
    id_ceo = findall(data.ceo .> 0.0)
    id = id_mwp ∩ id_ceo
    df = data[id, [:pr, :mwp]]
    d = dstat(df.pr, df.mwp);
    p = dplot(d)
    fig07 = plot(p.all)
    savefig("figure07.pdf")
    println("figure07.pdf")
    display(d.summary)
    println("\n")
end



## 7.2 Example 4: Gluing PQD and NQD 
## Figures 8,9,10

# Figure 8
begin
    archivo = "example4dataset.csv"
    data = DataFrame(CSV.File(archivo))
    d = dstat(data.ir_min, data.contrast)
    p = dplot(d)
    fig08 = plot(p.all)
    savefig("figure08.pdf")
    println("figure08.pdf")
    display(d.summary)
    println("\n")
end

# Gluing in u ≈ 0.8
begin
    iordx = sortperm(d.x)
    xy = d.xy[iordx, :]
    xyA = xy[1:819, :] # 0.8 × 1024 obs ≈ 819
    xyB = xy[819:end, :]
    dA = dstat(xyA[:, 1], xyA[:, 2])
    pA = dplot(dA)
    dB = dstat(xyB[:, 1], xyB[:, 2])
    pB = dplot(dB)
end;

# Figure 9
begin
    # σ = 0.760021  ρ = 0.760002   r = 0.649281
    fig09 = plot(pA.all)
    savefig("figure09.pdf")
    println("figure09.pdf")
    display(dA.summary)
    println("\n")
end

# Figure 10 
begin
    fig10 = plot(pB.all)
    savefig("figure10.pdf")
    println("figure10.pdf")
    display(dB.summary)
    println("\n")
end


## Example 5: Apparently independent

# Figure 11
begin
    archivo = "example5dataset.csv"
    data = DataFrame(CSV.File(archivo))
    n = 1_000
    N = collect(1:nrow(data))
    iSample = sample(N, n, replace = false)
    d = dstat(data.tempo[iSample], data.acousticness[iSample])
    p = dplot(d)
    # σ = 0.198438  ρ = -0.19438  r = -0.197088
    fig11 = plot(p.all)
    savefig("figure11.pdf")
    println("figure11.pdf")
    display(d.summary)
    println("\n")
end

println("*** End of generating figures.", "\n")