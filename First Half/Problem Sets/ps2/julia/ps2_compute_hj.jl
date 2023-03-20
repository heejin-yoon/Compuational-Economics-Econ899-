using Parameters, Plots, Printf, Weave

rt = pwd()

include(rt * "/First Half/Problem Sets/ps2/julia/ps2_model_hj.jl") #import the functions that solve our growth model

## Solve the model

prim, res = Initialize() #initialize primitive and results structs
elapse = @elapsed Solve_model(prim, res) #solve the model!
println(" ")
@printf("It took %0.3f seconds to solve the model using Julia.", float(elapse))
println(" ")
println(" ")

## Upper bound of asset

n = sum(res.pol_func[:, 1] .> prim.a_grid[:, 1])
a_ub = prim.a_grid[n, 1]
@printf("ā for employed agent is %0.3f.", float(a_ub))
println(" ")
println(" ")

## Plot figures

# Plot value functions
Plots.plot(prim.a_grid, res.val_func, title="Value Functions", labels=["Employed" "Unemployed"])
Plots.savefig("ps2_value_functions.png")
println("Value functions are saved.")
println(" ")

# Plot policy functions
Plots.plot(prim.a_grid, [res.pol_func prim.a_grid], title="Policy Functions", labels=["Employed" "Unemployed" "45° Line"])
plot!([a_ub], seriestype="vline", linestyle=:dash, linecolor="grey", labels="ā = 1.048")
Plots.savefig("ps2_policy_functions.png")
println("Policy functions are saved.")
println(" ")

# Plot asset distribution
Plots.plot(prim.a_grid, res.μ, title="Asset Distribution", labels=["Employed" "Unemployed"])
Plots.savefig("ps2_asset_distributions.png")
println("Asset Distributions are saved.")
println(" ")

## Wealth Distribution

w, cdf_w, cum_w = w_dist(prim, res)

# Plot Wealth distribution
Plots.plot(prim.a_grid, w, title="Wealth Distribution", labels=["Employed" "Unemployed"])
Plots.savefig("ps2_wealth_distributions.png")
println("Wealth Distributions are saved.")
println(" ")

# Lorenz Curve
Plots.plot(cdf_w, [cum_w cdf_w], title="Lorenz Curve", labels=["45° Line" "Lorenz Curve"])
Plots.savefig("ps2_lorenz_curve.png")
println("Lorenz Curve is saved.")
println(" ")

# Gini Coefficient
gini_coeff = gini(prim, cdf_w, cum_w)
@printf("Gini Coefficient is %0.3f.", float(gini_coeff))
println(" ")
println(" ")


## Welfare Analysis

Welfare_fb, lambda, pro_fb = Wfb_lambda(prim, res)

# Plot λ(a,s)
Plots.plot(prim.a_grid, lambda, title="λ(a, s)", labels=["Employed" "Unemployed"])
Plots.savefig("ps2_lambda.png")
println("Lambda is saved.")
println(" ")

Welfare_inc = W_inc(prim, res)
Welfare_gain = Welfare_fb - Welfare_inc

@printf("Welfare gain of moving from incomplete to complete financial marekt is %0.3f (%0.3f ⇒ %0.3f).", float(Welfare_gain), float(Welfare_inc), float(Welfare_fb))
println(" ")
println(" ")
@printf("The fraction of the population who favor moving into complete markets is %0.3f.", float(pro_fb))
println(" ")
println(" ")
println("All done!")
println(" ")

##

weave("ps2_model_hj.jl", doctype="md2pdf", out_path="weave")
