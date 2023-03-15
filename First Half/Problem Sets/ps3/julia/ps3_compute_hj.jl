
## Open Packages and Files

using Parameters, Plots, Setfield                                                # import the libraries we want
include("ps3_model_new.jl");                                                     # import the functions that solve our growth model
prim = Primitives()
res = Initialize()

## Exercise 1

# solve dynamic programming problem

prim = Primitives()
res = Initialize()                                                               # initialize primitive and results structs
@elapsed solve_HH_problem(prim, res)

# draw value function

plot(prim.a_grid, res.val_func[50, :, 1], labels = "", legend=:topleft)
savefig("val_func_50.png")

# draw policy function

plot([prim.a_grid prim.a_grid prim.a_grid], [res.pol_func[20, :, 1] res.pol_func[20, :, 2] prim.a_grid], labels = ["High" "Low" "45° Line"], legend=:topleft)
savefig("pol_func_20.png")

# draw savings functnion

plot([prim.a_grid prim.a_grid], [res.pol_func[20, :, 1].-prim.a_grid res.pol_func[20, :, 2].-prim.a_grid], labels = ["High" "Low"], legend=:topleft)
savefig("sav_func_20.png")

## Exercise 2

prim = Primitives()
res = Initialize()

@elapsed  Fj = μ_distribution(prim, res)

## Exercise 3

# 3.1 benchmark (with or without social security)

prim = Primitives()
res = Initialize()
@elapsed solve_model(prim, res)
bm_ss = welfare_analysis(prim, res)

println("★ Benchmark Model with Social Security ★")
println("***********************************")

prim = Primitives()
res = Initialize()
prim = @set prim.θ = 0
@elapsed solve_model(prim, res)
bm_no_ss = welfare_analysis(prim, res)

println("★ Benchmark Model Without Social Security ★")
println("***********************************")

# 3.2  no risk (with or without social security)

prim = Primitives()
res = Initialize()
prim = @set prim.z = [0.5, 0.5]
prim = @set prim.e = prim.η*prim.z'

@elapsed solve_model(prim, res)
nr_ss = welfare_analysis(prim, res)

println("★ No Risk Model with Social Security ★")
println("***********************************")

prim = Primitives()
res = Initialize()
prim = @set prim.z = [0.5, 0.5]
prim = @set prim.e = prim.η*prim.z'
prim = @set prim.θ = 0

@elapsed solve_model(prim, res)
nr_no_ss = welfare_analysis(prim, res)

println("★ No Risk Model Without Social Security ★")
println("***********************************")

# 3.3  exogeneous labor (with or without social security)

prim = Primitives()
res = Initialize()
prim = @set prim.γ = 1

@elapsed solve_model(prim, res)
el_ss = welfare_analysis(prim, res)

println("★ Exogenous Labor Model with Social Security ★")
println("***********************************")

prim = Primitives()
res = Initialize()
prim = @set prim.γ = 1
prim = @set prim.θ = 0

@elapsed solve_model(prim, res)
el_no_ss = welfare_analysis(prim, res)

println("★ Exogenous Labor Model Without Social Security ★")
println("***********************************")

## Finish

println("All done!")
