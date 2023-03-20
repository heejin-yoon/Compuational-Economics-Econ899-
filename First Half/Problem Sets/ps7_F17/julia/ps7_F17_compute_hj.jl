using Parameters, Plots, Interpolations, Optim, Printf

rt = pwd()

include(rt * "/First Half/Problem Sets/ps7_F17/julia/ps7_F17_model_hj.jl")

##

prim, res = Initialize()
res_linear = solve_model(prim, res, "Linear")

prim, res = Initialize()
res_cubic = solve_model(prim, res, "Cubic")

##

val_func_interp_linear = interpolate(res_linear.val_func, BSpline(Linear()))
pol_func_interp_linear = interpolate(res_linear.pol_func, BSpline(Linear()))
val_func_interp_cubic = interpolate(res_cubic.val_func, (BSpline(Cubic(Line(OnGrid()))), BSpline(Linear())))
pol_func_interp_cubic = interpolate(res_cubic.pol_func, (BSpline(Cubic(Line(OnGrid()))), BSpline(Linear())))

k_ss_index, K_ss_index = ss_index(prim, res)

# Value functions
plot(prim.k_grid, val_func_interp_linear[:, K_ss_index], title="Value function with linear` interpolation at K = Kₛₛ", xlabel="k", labels="")
plot(prim.K_grid, val_func_interp_linear[k_ss_index, :], title="Value function with linear interpolation at k = kₛₛ", xlabel="K", labels="")
plot(prim.k_grid, val_func_interp_cubic[:, K_ss_index], title="Value function with cubic interpolation at K = Kₛₛ", xlabel="k", labels="")
plot(prim.K_grid, val_func_interp_cubic[k_ss_index, :], title="Value function with cubic interpolation at k = kₛₛ", xlabel="K", labels="")

# Policy functions
plot(prim.k_grid, pol_func_interp_linear[:, K_ss_index], title="Policy function with linear interpolation at K = Kₛₛ", xlabel="k", labels="")
plot(prim.K_grid, pol_func_interp_linear[k_ss_index, :], title="Policy function with linear interpolation at k = kₛₛ", xlabel="K", labels="")
plot(prim.k_grid, pol_func_interp_cubic[:, K_ss_index], title="Policy function with cubic interpolation at K = Kₛₛ", xlabel="k", labels="")
plot(prim.K_grid, pol_func_interp_cubic[k_ss_index, :], title="Policy function with cubic interpolation at k = kₛₛ", xlabel="K", labels="")

##

println("All done.")
