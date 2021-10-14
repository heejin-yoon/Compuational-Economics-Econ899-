# ------------------------------------------------------------------------------
# Author: Heejin
# Huggett (1993, JEDC)
# September 20, 2021
# ps2_model.jl
# ------------------------------------------------------------------------------

using Parameters, Plots                                                          # import the libraries we want

include("ps2_model.jl");                                                         # import the functions that solve our growth model

prim, res = Initialize()                                                         # initialize primitive and results structs

@unpack a_grid, α, β = prim

Solve_model(prim, res)

Plots.plot(a_grid, res.val_func, labels = ["Employed" "Unemployed"])             # plot value function
savefig("value_function.png")

Plots.plot(a_grid, [res.pol_func a_grid],                                        # plot policy function
     labels = ["Employed" "Unemployed" "45° Line"],
     legend=:topleft)
savefig("policy_function.png")

a_hat_e = a_grid[argmin(abs.(res.pol_func[:, 1] - a_grid))]                      # Dean's a_hat

plot(a_grid, res.μ, labels = ["Employed" "Unemployed"], legend=:topleft)         # bond holdings distribution
savefig("bond_distribution.png")

wealth = Wealth_distribution(prim, res)                                          # calculate wealth distribution
plot(a_grid, wealth, labels = ["Employed" "Unemployed"], legend=:topleft)        # plot wealth distribution
savefig("wealth_distribution.png")

wealth_spike = a_grid[argmax(wealth)]                                            # biggest wealth spike

lorenz = Lorenz_curve(prim, wealth);
plot([lorenz[:,1] lorenz[:,1]], [lorenz[:,1] lorenz[:,2]], labels = ["45° Line" "Lorenz Curve"],
     legend=:topleft)
savefig("lorenz_curve.png")

gini_coefficient = Gini(lorenz)

welfare_fb = Welfare_fb(prim)                                                    # welfare in complete market

λ = calculate_λ(prim, res, welfare_fb)
plot(a_grid, λ, labels = ["Employed" "Unemployed"])
savefig("lambda.png")

welfare_incomplete = sum(res.μ .* res.val_func)                                  # welfare in incomplete market

welfare_gain = sum(res.μ .* λ)

fraction_better = sum((λ .>= 0) .*res.μ)

println("All done!")
################################
