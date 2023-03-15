using Parameters, Plots                                                          # import the libraries we want

include("ps3_model.jl");                                                         # import the functions that solve our growth model

prim = Primitives()
prim, res = Initialize(prim, 0.11, [3.0, 0.5], 0.42)                                                         # initialize primitive and results structs
@elapsed solve_HH_problem(prim, res)
@elapsed asset_dist(prim, res)
plot(prim.a_grid, res.val_func[50, :, 1], labels = "", legend=:bottomright)
savefig("value_function_50.png")

plot([prim.a_grid prim.a_grid prim.a_grid], [res.pol_func[20, :, :] prim.a_grid],
     labels = ["High" "Low" "45° Line"],
     legend=:bottomright)

savefig("policy_function_20.png")

# Benchmark
@elapsed bm_ss = solve_model(prim, res)  # converges in ~9 iterations
@elapsed bm_no_ss = solve_model(θ = 0.0)  # converges in ~11 iterations

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
