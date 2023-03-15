using Parameters, Plots, Weave, Printf

cd("C:/Users/hyoon76/OneDrive - UW-Madison/3.Wisconsin/2021 Fall/Econ 899/Problem Sets/ps1/julia/")
include("ps1_model_hj.jl") #import the functions that solve our growth model

## Q1

prim, res = Initialize() #initialize primitive and results structs
elapse = @elapsed Solve_model(prim, res) #solve the model!
@printf("It took %0.3f seconds to run the model using Julia.", float(elapse))
println("")
println("")

## Q2

# Plot value functions
Plots.plot(prim.k_grid, res.val_func, title="Value Function", labels = ["High Productivity" "Low Productivity"])
Plots.savefig("ps1_value_functions.png")
println("Value functions are saved.")
println(" ")

## Q3

# Plot policy functions
Plots.plot(prim.k_grid, res.pol_func, title="Policy Functions", labels = ["High Productivity" "Low Productivity"])
Plots.savefig("ps1_policy_functions.png")
println("Policy functions are saved.")
println(" ")

# Plot saving functions
saving_func = res.pol_func .- prim.k_grid
Plots.plot(prim.k_grid, saving_func, title="Saving Functions", labels = ["High Productivity" "Low Productivity"])
Plots.savefig("ps1_saving_functions.png")
println("Saving functions are saved.")
println(" ")

println("All done!")
println(" ")

##
# weave("ps1_model_hj.jl", doctype = "md2pdf", out_path = "weave")
