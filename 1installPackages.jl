### Visual analysis of bivariate dependence between continuous random variables
### Authors: Arturo Erdely and Manuel Rubio-Sánchez
### Version date: 2024-06-23

### INSTALL/LOAD REQUIRED JULIA PACKAGES

using Pkg

begin
    paquete = ["CSV", "DataFrames", "Distributions", "Plots", "StatsPlots" ]
    for p in paquete
        println("*** Installing package: ", p)
        Pkg.add(p)
    end
    println("*** End of package list.")
end

println("\n")