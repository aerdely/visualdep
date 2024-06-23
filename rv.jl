## Random variable probability models
## Author: Arturo Erdely
## Version date: 2024-02-11
"""
Random variable probability models as tuple objects:

- Exponential (location-dispersion family)
- Kumaraswamy (2 and 4-parameter versions, and mixture of two)
- Pareto
- Student (standard and location-dispersion family)
- Uniform (continuous)

> Dependence on external package: Distributions (just for Student)
"""

include("exponential.jl")
include("kumaraswamy.jl")
include("pareto.jl")
include("student.jl")
include("uniform.jl")

rvList = ["rvExponential", 
          "rvKuma", "rvKuma4", "rvKumaV", "rvKumaV4", "rvKumaMix",
          "rvPareto", 
          "rvStudent", "rvStudent3", "rvStudentMix",
          "rvUniform"
]

function rvIndex()
    for model ∈ rvList
        println(model)
    end
end

; # end of file