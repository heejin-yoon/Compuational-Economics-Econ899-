using Parameters, DataFrames, StatFiles, LinearAlgebra, Statistics, Plots, Optim, Printf, .Threads

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps3/ps3_model.jl")

prim, res = Initialize()

## Q1. Invert Demand Function using Two Algorithms

δ_Contraction = get_δ(1985, 0.6, "Contraction")
δ_Newton = get_δ(1985, 0.6, "Newton")


## Q2. 

λ_1step, minval_1step = solve_GMM("Newton", "1step") # (0.6194078867223469, 234.76077311621532)


## Q3. 

λ_2step, minval_2step = solve_GMM("Newton", "2step") # (0.6920209245379816, 163.13409857517448)


##

λ_grid = collect(0.0:0.05:1.0)
objftn_1step = zeros(length(λ_grid))
objftn_2step = zeros(length(λ_grid))

for λ_index = 1:length(λ_grid)
    println(λ_index)
    objftn_1step[λ_index] = object_function(λ_grid[λ_index], "Newton", "1step")
    objftn_2step[λ_index] = object_function(λ_grid[λ_index], "Newton", "2step")
end

plot(λ_grid, [objftn_1step, objftn_2step], label = ["1-Step GMM" "2-Step GMM"], xlabel = "λ", ylabel = "Objective Function", lw=1.5)
plot!([λ_1step; λ_2step], [minval_1step; minval_2step], seriestype=:scatter, label="Optimum", mc=:gray, ms=3, ma=0.8)
savefig(rt * "/Second Half/Problem Sets/ps3/gmm_objectftn.png")