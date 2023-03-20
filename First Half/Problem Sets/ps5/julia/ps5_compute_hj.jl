using Interpolations, Plots, Parameters, DataFrames, Random, Distributions, GLM, Optim, Printf

rt = pwd()

include(rt * "/First Half/Problem Sets/ps5/julia/ps5_model_hj.jl")

##

prim, res = Initialize()
elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the whole model.\n\n", float(elapse))

## Plot figures at K = K_ss

K_ss_index = get_index(prim.K_ss, prim.K_grid)

val_func_interp = interpolate(res.val_func, BSpline(Linear()))
pol_func_interp = interpolate(res.pol_func, BSpline(Linear()))

plot(prim.k_grid, [val_func_interp[:, 1, K_ss_index, 1] val_func_interp[:, 2, K_ss_index, 1] val_func_interp[:, 1, K_ss_index, 2] val_func_interp[:, 2, K_ss_index, 2]], title="Value functions at K = Kₛₛ", labels=["Employed, z = zʰ" "Unemployed, z = zʰ" "Employed, z = zˡ" "Unemployed, z = zˡ"], xlabel="k", ylabel="value function", legend=:bottomright)
savefig("val_func.png")
println("Value functions are saved.\n")

plot(prim.k_grid, [pol_func_interp[:, 1, K_ss_index, 1] pol_func_interp[:, 2, K_ss_index, 1] pol_func_interp[:, 1, K_ss_index, 2] pol_func_interp[:, 2, K_ss_index, 2]], title="Policy functions at K = Kₛₛ", labels=["Employed, z = zʰ" "Unemployed, z = zʰ" "Employed, z = zˡ" "Unemployed, z = zˡ"], xlabel="k", ylabel="saving function", legend=:bottomright)
savefig("pol_func.png")
println("Policy functions are saved.\n")

plot(prim.k_grid, [pol_func_interp[:, 1, K_ss_index, 1] - prim.k_grid pol_func_interp[:, 2, K_ss_index, 1] - prim.k_grid pol_func_interp[:, 1, K_ss_index, 2] - prim.k_grid pol_func_interp[:, 2, K_ss_index, 2] - prim.k_grid], title="Saving functions at K = Kₛₛ", labels=["Employed, z = zʰ" "Unemployed, z = zʰ" "Employed, z = zˡ" "Unemployed, z = zˡ"], xlabel="k", ylabel="saving function")
savefig("sav_func.png")
println("Saving functions are saved.\n")

plot_path(prim, res)
savefig("K_movement.png")
println("K movements are saved.\n")

##

println("All done.")
