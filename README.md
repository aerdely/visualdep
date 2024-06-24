# Visual analysis of bivariate dependence between continuous random variables

> Authors: Arturo Erdely & Manuel Rubio-Sánchez
>
> Preprint: [arXiv](https://arxiv.org/abs/2404.00820)
>
> ### Instructions for reproducibility

1. Download and install the [Julia](https://julialang.org/downloads/) programming language.
2. Download the code files clicking in the green button `<> Code` of this GitHub repository and `DownloadZIP`. Unzip the downloaded file in a working directory of your election.
3. Open the `Julia` terminal and change to the working directory where you unzipped the files. You may do this by defining a string variable `path` with the path to the files directory and then execute in the terminal `cd(path)`. For example, in the operating system *Windows* it may look something like:
   ```julia
   path = "D:/MyFiles/visualdep"
   cd(path)
   readdir()
   ```
4. Install the required packages by executing the following command in the `Julia` terminal. This may take a while:
   ```julia
   include("1installPackages.jl")
   ```
5. Load the required `Julia` code by executing the following command in the `Julia` terminal:
   ```julia
   include("2loadFunctions.jl")
   ```
6. Generate the figures and calculations by executing the following command in the `Julia` terminal:
   ```julia
   include("3generateFigures.jl")
   ```
   Figure 3 takes several minutes, be patient. Don't worry about the warning messages from the data for figures 7 through 11.
