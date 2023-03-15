# ------------------------------------------------------------------------------
# Author: Heejin
# Hopenhayn and Rogerson (1993, JPE)
# October 21, 2021
# ps6_model.jl
# ------------------------------------------------------------------------------
using Parameters, Plots, Setfield                                                          # import the libraries we want

include("ps6_model.jl");                                                         # import the functions that solve our growth model

prim, res = Initialize()

@elapsed standard = solve_model(prim, res)

prim, res = Initialize()
prim = @set prim.α = 1
@elapsed α_1 = solve_model(prim, res)

prim, res = Initialize()
prim = @set prim.α = 2
@elapsed α_2 = solve_model(prim, res)

plot([standard.exit_func α_1.exit_func α_2.exit_func],
             label = ["Standard" "TV1 Shocks (α = 1)" "TV1 Shocks (α = 2)"],
             title = "Decision Rules of Exit", legend = :bottomright)
savefig("exit_func1.png")


prim, res = Initialize()
prim = @set prim.cf = 15.0
@elapsed standard = solve_model(prim, res)

prim, res = Initialize()
prim = @set prim.cf = 15.0
prim = @set prim.α = 1
@elapsed α_1 = solve_model(prim, res)

prim, res = Initialize()
prim = @set prim.cf = 15.0
prim = @set prim.α = 2
@elapsed α_2 = solve_model(prim, res)

plot([standard.exit_func α_1.exit_func α_2.exit_func],
             label = ["Standard" "TV1 Shocks (α = 1)" "TV1 Shocks (α = 2)"],
             title = "Decision Rules of Exit", legend = :bottomright)
savefig("exit_func2.png")
