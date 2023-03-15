using Parameters, Plots, Printf, Setfield, DelimitedFiles

cd("C:/Users/hyoon76/OneDrive - UW-Madison/5.Miscellaneous/CompEcon Practice/ps4/")
include("ps4_model_stationary_hj.jl");
include("ps4_model_transition_hj.jl");

## Exercise 1.

println("Recall problem set #3")
println("# benchmark with social security")

prim, res = Initialize()

elapse = @elapsed solve_model(prim, res)
println(" ")
@printf("It took %0.3f seconds to solve the model using Julia.", float(elapse))
println(" ")
println(" ")
bm_ss = welfare_analysis(prim, res)

println("***************************************")
println("")
println("★ Benchmark Model with Social Security ★")
println("")
@printf("Total welfare: %0.3f.", float(bm_ss.W))
println("")
@printf("CV: %0.3f.", float(bm_ss.CV))
println("")
println("")
println("***************************************")

println("# benchmark without social security")

prim, res = Initialize()
prim = @set prim.θ = 0

elapse = @elapsed solve_model(prim, res)
println(" ")
@printf("It took %0.3f seconds to solve the model using Julia.", float(elapse))
println(" ")
println(" ")

bm_wo_ss = welfare_analysis(prim, res)

println("***************************************")
println("")
println("★ Benchmark Model Without Social Security ★")
println("")
@printf("Total welfare: %0.3f.", float(bm_wo_ss.W))
println("")
@printf("CV: %0.3f.", float(bm_wo_ss.CV))
println("")
println("")
println("***************************************")

## Compute transition path

prim, res = Initialize()

elapse = @elapsed path, T = solve_model_trans(prim, res, 1, 30)
println(" ")
@printf("It took %0.3f seconds to solve the HH problem.", float(elapse))
println(" ")
println(" ")

## Draw figures

# interest rate path

plot(collect(1:T), [path.r_path repeat([bm_ss.r], T) repeat([bm_wo_ss.r], T)],
             label = ["Interest rate path" "Stationary rate w/ SS" "Stationary rate w/o SS"],
             title = "Interest Rate Transition Path", legend = :bottomright)
savefig("exercise1_interest_rate_path.png")
println("Interest rate path is saved.")
println(" ")

# wage path

plot(collect(1:T), [path.w_path repeat([bm_ss.w], T) repeat([bm_wo_ss.w], T)],
              label = ["Wage path" "Stationary wage w/ SS" "Stationary wage w/o SS"],
              title = "Wage Transition Path", legend = :bottomright)
savefig("exercise1_wage_path.png")
println("Wage path is saved.")
println(" ")

# aggregate capital path

plot(collect(1:T), [path.K_path repeat([bm_ss.K], T) repeat([bm_wo_ss.K], T)],
             label = ["Aggregate capital path" "Stationary capital w/ SS" "Stationary capital w/o SS"],
             title = "Capital Transition Path", legend = :bottomright)
savefig("exercise1_aggregate_capital_path.png")
println("Aggregate capital path is saved.")
println(" ")

# aggregate labor path

plot(collect(1:1:T), [path.L_path repeat([bm_ss.L], T) repeat([bm_wo_ss.L], T)],
          label = ["Aggregate labor path" "Stationary labor w/ SS" "Stationary labor w/o SS"],
          title = "Labor Transition Path", legend = :bottomright)
savefig("exercise1_aggregate_labor_path.png")
println("Aggregate labor path is saved.")
println(" ")

## Welfare analysis

EV_j, EV_j_counterfact, EV, EV_counterfact = Equivalent_variation(prim, res, path)

plot(collect(20:82), [EV_j[1:63], EV_j_counterfact[1:63], repeat([1], 63)], labels = ["With Transition" "Without Transition" "EV = 1"],
          title = "EV by Age", legend = :bottomleft)
savefig("exercise1_EV.png")
println("Average welfare effect within each age cohorts are saved.")
println(" ")

vote_share = Vote_share(EV)
vote_share_counterfact = Vote_share(EV_counterfact)
diff_vote_share = vote_share_counterfact - vote_share
@printf("With transition, %0.2f percent of population will vote for the reform.", float(vote_share*100))
println("")
println("")
@printf("Without transition, %0.2f percent of population will vote for the reform.", float(vote_share_counterfact*100))
println("")
println("")
@printf("In sum, %0.2f percent of vote share is over-estimated if we do not consider transition path.", float(diff_vote_share*100))
println("")
println("")

## Exercise 2.

prim, res = Initialize()

elapse = @elapsed path2, T2 = solve_model_trans(prim, res, 20, 50)
println(" ")
@printf("It took %0.3f seconds to solve the HH problem.", float(elapse))
println(" ")
println(" ")

## Draw figures

# interest rate path

plot(collect(1:T2), [path2.r_path repeat([bm_ss.r], T2) repeat([bm_wo_ss.r], T2)],
             label = ["Interest rate path" "Stationary rate w/ SS" "Stationary rate w/o SS"],
             title = "Interest Rate Transition Path", legend = :bottomright)
savefig("exercise2_interest_rate_path.png")
println("Interest rate path is saved.")
println(" ")

# wage path

plot(collect(1:T2), [path2.w_path repeat([bm_ss.w], T2) repeat([bm_wo_ss.w], T2)],
              label = ["Wage path" "Stationary wage w/ SS" "Stationary wage w/o SS"],
              title = "Wage Transition Path", legend = :bottomright)
savefig("exercise2_wage_path.png")
println("Wage path is saved.")
println(" ")

# aggregate capital path

plot(collect(1:T2), [path2.K_path repeat([bm_ss.K], T2) repeat([bm_wo_ss.K], T2)],
             label = ["Aggregate capital path" "Stationary capital w/ SS" "Stationary capital w/o SS"],
             title = "Capital Transition Path", legend = :bottomright)
savefig("exercise2_aggregate_capital_path.png")
println("Aggregate capital path is saved.")
println(" ")

# aggregate labor path

plot(collect(1:T2), [path2.L_path repeat([bm_ss.L], T2) repeat([bm_wo_ss.L], T2)],
          label = ["Aggregate labor path" "Stationary labor w/ SS" "Stationary labor w/o SS"],
          title = "Labor Transition Path", legend = :bottomright)
savefig("exercise2_aggregate_labor_path.png")
println("Aggregate labor path is saved.")
println(" ")

## Welfare analysis

EV_j2, EV_j_counterfact2, EV2, EV_counterfact2 = Equivalent_variation(prim, res, path2)

plot(collect(20:82), [EV_j[1:63], EV_j2[1:63], repeat([1], 63)], labels = ["Policy: t = 1" "Policy: t = 21" "EV = 1"],
          title = "EV by Age", legend = :bottomleft)
savefig("exercise2_EV.png")
println("Average welfare effect within each age cohorts are saved.")
println(" ")

vote_share2 = Vote_share(EV2)

@printf("%0.2f percent of population will vote for the reform.", float(vote_share2*100))
println("")
println("")
@printf("The fraction of population who support the reform substantially increases from %0.2f to %0.2f percent.", float(vote_share*100), float(vote_share2*100))
println("")
println("")

##

println("All jobs are done!")
