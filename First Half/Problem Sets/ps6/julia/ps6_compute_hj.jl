using Parameters, Plots, Printf, DelimitedFiles, DataFrames, Setfield

cd("C:/Users/hyoon76/OneDrive - UW-Madison/5.Miscellaneous/CompEcon Practice/ps6/")
include("ps6_model_hj.jl");

## Task 1.

prim, res = Initialize()
standard = solve_model(prim, res, 0.0)

prim, res = Initialize()
tv1_1 = solve_model(prim, res, 1.0)

prim, res = Initialize()
tv1_2 = solve_model(prim, res, 2.0)

## Task 2.

res_standard = result_table(prim, standard)
res_tv1_1 = result_table(prim, tv1_1)
res_tv1_2 = result_table(prim, tv1_2)
data_summary = DataFrame(item=["Price Level", "Mass of Incumbents", "Mass of Entrants", "Mass of Exits", "Aggregate Labor", "Labor of Incumbents", "Labor of Entrants", "Fraction of Labor in Entrants"], Standard = res_standard, TV1_1 = res_tv1_1, TV1_2 = res_tv1_2)
writedlm("data_summary.csv", Iterators.flatten(([names(data_summary)], eachrow(data_summary))), ',')

## Task 3.

plot([standard.exit_func tv1_1.exit_func tv1_2.exit_func],
             label = ["Standard" "TV1 Shocks (α = 1)" "TV1 Shocks (α = 2)"],
             title = "Decision Rules of Exit", legend = :bottomright)
savefig("exit_func1.png")

## Task 4.

prim, res = Initialize()
prim = @set prim.c_f = 15.0
standard = solve_model(prim, res, 0.0)

prim, res = Initialize()
prim = @set prim.c_f = 15.0
tv1_1 = solve_model(prim, res, 1.0)

prim, res = Initialize()
prim = @set prim.c_f = 15.0
tv1_2 = solve_model(prim, res, 2.0)

res_standard = result_table(prim, standard)
res_tv1_1 = result_table(prim, tv1_1)
res_tv1_2 = result_table(prim, tv1_2)
data_summary2 = DataFrame(item=["Price Level", "Mass of Incumbents", "Mass of Entrants", "Mass of Exits", "Aggregate Labor", "Labor of Incumbents", "Labor of Entrants", "Fraction of Labor in Entrants"], Standard = res_standard, TV1_1 = res_tv1_1, TV1_2 = res_tv1_2)
writedlm("data_summary2.csv", Iterators.flatten(([names(data_summary2)], eachrow(data_summary2))), ',')

plot([standard.exit_func tv1_1.exit_func tv1_2.exit_func],
             label = ["Standard" "TV1 Shocks (α = 1)" "TV1 Shocks (α = 2)"],
             title = "Decision Rules of Exit", legend = :bottomright)
savefig("exit_func2.png")

##

println("All done!")
