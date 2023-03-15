##
using Parameters, Plots, Printf, Weave

cd("C:/Users/hyoon76/OneDrive - UW-Madison/5.Miscellaneous/CompEcon Practice/ps2b/julia/")
include("ps2b_model_hj.jl") #import the functions that solve our growth model

## Solve the model

prim, res = Initialize() #initialize primitive and results structs
elapse = @elapsed Solve_model(prim, res) #solve the model!
println(" ")
@printf("It took %0.3f seconds to solve the model using Julia.", float(elapse))
println(" ")
println(" ")

## Plot figures

# Plot value functions
Plots.plot(prim.a_grid, res.val_func[:, :, 1], title="Value Functions (h = 0)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_value_functions_0.png")
Plots.plot(prim.a_grid[prim.a_index_zero:prim.na], res.val_func[prim.a_index_zero:prim.na, :, 2], title="Value Functions (h = 1)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_value_functions_1.png")
println("Value functions are saved.")
println(" ")

Plots.plot(prim.a_grid, res.pol_func[:, :, 1], title="Policy Functions (h = 0)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_policy_functions_0.png")
Plots.plot(prim.a_grid[prim.a_index_zero:prim.na], res.pol_func[prim.a_index_zero:prim.na, :, 2], title="Policy Functions (h = 1)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_policy_functions_1.png")
println("Policy functions are saved.")
println(" ")

Plots.plot(prim.a_grid[1:prim.a_index_zero], res.def_func[1:prim.a_index_zero, :], title="Default Functions", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_default_functions.png")
println("Default functions are saved.")
println(" ")

Plots.plot(prim.a_grid, res.μ[:, :, 1], title="μ Distributions (h = 0)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_mu_distributions_0.png")
Plots.plot(prim.a_grid, res.μ[:, :, 2], title="μ Distributions (h = 1)", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_mu_distributions_1.png")
println("Mu Distributions are saved.")
println(" ")


## Wealth Distribution

w, cdf_w, cum_w = w_dist(prim, res)

# Plot Wealth distribution
Plots.plot(prim.a_grid, w, title="Wealth Distribution", labels = ["Employed" "Unemployed"])
Plots.savefig("ps2_wealth_distributions.png")
println("Wealth Distributions are saved.")
println(" ")

# Lorenz Curve
Plots.plot(cdf_w, [cum_w cdf_w], title="Lorenz Curve", labels = ["45° Line" "Lorenz Curve"])
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
Plots.plot(prim.a_grid, lambda, title="λ(a, s)", labels = ["Employed" "Unemployed"])
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

weave("ps2_model_hj.jl", doctype = "md2pdf", out_path = "weave")
