
## Open Packages and Files

using Parameters, Plots, Setfield
include("ps4_model_stationary_hj.jl")

## Recall Problem Set 3

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

## Exercise 1.

include("ps4_model_transition_hj.jl");

@elapsed path, T = solve_model_transition(prim, res, 1)

plot([path.r_path repeat([bm_ss.r], T) repeat([bm_no_ss.r], T)],
             label = ["Interest rate path" "Stationary rate w/ SS" "Stationary rate w/o SS"],
             title = "Interest Rate Transition Path", legend = :bottomright)
savefig("interest_rate_path.png")

plot([path.w_path repeat([bm_ss.w], T) repeat([bm_no_ss.w], T)],
              label = ["Wage path" "Stationary wage w/ SS" "Stationary wage w/o SS"],
              title = "Wage Transition Path", legend = :bottomright)
savefig("wage_path.png")

plot([path.K0_path repeat([bm_ss.K0], T) repeat([bm_no_ss.K0], T)],
             label = ["Aggregate capital path" "Stationary capital w/ SS" "Stationary capital w/o SS"],
             title = "Capital Transition Path", legend = :bottomright)
savefig("aggregate_capital_path.png")

plot([path.L0_path repeat([bm_ss.L0], T) repeat([bm_no_ss.L0], T)],
          label = ["Aggregate labor path" "Stationary labor w/ SS" "Stationary labor w/o SS"],
          title = "Labor Transition Path", legend = :bottomright)
savefig("aggregate_labor_path.png")

EVⱼ, EV = Equivalent_variation(prim, res, path)

plot([EVⱼ, prim.N], label = false,
          title = "EV by Age")
savefig("EV.png")

share = Vote_share(EV)

## Exercise 2.

@elapsed path2, T2 = solve_model_transition(prim, res, 20)

plot([path2.r_path repeat([bm_ss.r], T2) repeat([bm_no_ss.r], T2)],
             label = ["Interest rate path" "Stationary rate w/ SS" "Stationary rate w/o SS"],
             title = "Interest Rate Transition Path", legend = :bottomright)
savefig("interest_rate_path2.png")

plot([path2.w_path repeat([bm_ss.w], T2) repeat([bm_no_ss.w], T2)],
              label = ["Wage path" "Stationary wage w/ SS" "Stationary wage w/o SS"],
              title = "Wage Transition Path", legend = :bottomright)
savefig("wage_path2.png")

plot([path2.K0_path repeat([bm_ss.K0], T2) repeat([bm_no_ss.K0], T2)],
             label = ["Aggregate capital path" "Stationary capital w/ SS" "Stationary capital w/o SS"],
             title = "Capital Transition Path", legend = :bottomright)
savefig("aggregate_capital_path2.png")

plot([path2.L0_path repeat([bm_ss.L0], T2) repeat([bm_no_ss.L0], T2)],
          label = ["Aggregate labor path" "Stationary labor w/ SS" "Stationary labor w/o SS"],
          title = "Labor Transition Path", legend = :bottomright)
savefig("aggregate_labor_path2.png")

EVⱼ2, EV2 = Equivalent_variation(prim, res, path2)

plot([EVⱼ2, prim.N], label = false,
          title = "EV by Age")
savefig("EV2.png")

share2 = Vote_share(EV2)

##

println("All jobs are done!")
