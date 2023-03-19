using Parameters, DataFrames, StatFiles, LinearAlgebra, Statistics, Plots, Optim, Printf, .Threads

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps3/ps3_model.jl")

prim, res = Initialize()

## Q1. Invert Demand Function using Two Algorithms

δ_Contraction = get_δ(1985, 0.6, "Contraction")
δ_Newton = get_δ(1985, 0.6, "Newton")


## Q2. 

λ_1step, minval_1step = solve_GMM("Newton", "1step")
# (0.6194078867223469, 234.76077311621532)


## Q3. 

λ_2step, minval_2step = solve_GMM("Newton", "2step")
# (0.6920209245379816, 163.13409857517448)