using Parameters, Plots, Printf, Setfield, DataFrames, DelimitedFiles

rt = pwd()

include(rt * "/First Half/Problem Sets/ps3/julia/ps3_model_hj.jl")

## Exercise 1

prim, res = Initialize()
elapse = @elapsed res.val_func, res.pol_func, res.lab_func = Bellman(prim, res)
@printf("\nIt took %0.3f seconds to solve the HH problem.\n\n", float(elapse))


## Draw figures

# Value function

plot(prim.a_grid, res.val_func[50, :, 1], title="Value Function at age 50", labels="", legend=:topleft)
savefig("val_func_50.png")
println("Value function at age 50 is saved.\n")


# Policy functions

plot(prim.a_grid, [res.pol_func[20, :, 1] res.pol_func[20, :, 2] prim.a_grid], title="Policy Functions at age 20", labels=["High" "Low" "45° Line"], legend=:topleft)
savefig("pol_func_20.png")
println("Policy function at age 20 is saved.\n")

# Saving functnions

plot(prim.a_grid, [res.pol_func[20, :, 1] .- prim.a_grid res.pol_func[20, :, 2] .- prim.a_grid], title="Saving Functions at age 20", labels=["High" "Low"], legend=:topright)
savefig("sav_func_20.png")
println("Saving function at age 20 is saved.\n")

## Exercise 2

elapse = @elapsed res.F = F_dist(prim, res)
μ = res.F[:, :, 1] * ones(prim.na) + res.F[:, :, 2] * ones(prim.na)
age = collect(1:1:prim.N)
@printf("\nIt took %0.3f seconds to calculate F distribution.\n\n", float(elapse))

plot(age, μ, title="Relative size of each cohort of age", labels="", legend=:topleft)
savefig("mu_dist.png")
println("μ distribution is saved.\n")

## Exercise 3: 3-1-1 benchmark with social security

prim, res = Initialize()

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n\n", float(elapse))
bm_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ Benchmark Model with Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(bm_ss.W))
@printf("CV: %0.3f.\n\n", float(bm_ss.CV))
println("***************************************")

## 3-1-2 benchmark without social security

prim, res = Initialize()
prim = @set prim.θ = 0

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n\n", float(elapse))

bm_wo_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ Benchmark Model Without Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(bm_wo_ss.W))
@printf("CV: %0.3f.\n\n", float(bm_wo_ss.CV))
println("***************************************")

## 3-2-1  No risk with social security

prim, res = Initialize()
prim = @set prim.z = [0.5, 0.5]
prim = @set prim.e = prim.η * prim.z'

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n\n", float(elapse))

nr_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ No Risk Model with Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(nr_ss.W))
@printf("CV: %0.3f.\n\n", float(nr_ss.CV))
println("***************************************")

## 3-2-2  No risk without social security

prim, res = Initialize()
prim = @set prim.z = [0.5, 0.5]
prim = @set prim.e = prim.η * prim.z'
prim = @set prim.θ = 0

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n\n", float(elapse))

nr_wo_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ No Risk Model Without Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(nr_wo_ss.W))
@printf("CV: %0.3f.\n\n", float(nr_wo_ss.CV))
println("***************************************")

## 3-3-1 exogeneous labor with social security

prim, res = Initialize()
prim = @set prim.γ = 1

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n\n", float(elapse))

el_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ Exogeneous Labor Model with Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(el_ss.W))
@printf("CV: %0.3f.\n\n", float(el_ss.CV))
println("***************************************")

## 3-3-2 exogeneous labor without social security

prim, res = Initialize()
prim = @set prim.γ = 1
prim = @set prim.θ = 0

elapse = @elapsed solve_model(prim, res)
@printf("\nIt took %0.3f seconds to solve the model using Julia.\n", float(elapse))

el_wo_ss = welfare_analysis(prim, res)

println("***************************************")
println("\n★ Exogeneous Labor Model Without Social Security ★\n")
@printf("Total welfare: %0.3f.\n", float(el_wo_ss.W))
@printf("CV: %0.3f.\n\n", float(el_wo_ss.CV))
println("***************************************")

##

data_summary = DataFrame(item=["capital K", "labor L", "wage w", "interest r", "pension benefit b", "total welfare W", "cv (wealth)"], BM_w_SS=[bm_ss.K, bm_ss.L, bm_ss.w, bm_ss.r, bm_ss.b, bm_ss.W, bm_ss.CV], BM_wo_SS=[bm_wo_ss.K, bm_wo_ss.L, bm_wo_ss.w, bm_wo_ss.r, bm_wo_ss.b, bm_wo_ss.W, bm_wo_ss.CV], NR_w_SS=[nr_ss.K, nr_ss.L, nr_ss.w, nr_ss.r, nr_ss.b, nr_ss.W, nr_ss.CV], NR_wo_SS=[nr_wo_ss.K, nr_wo_ss.L, nr_wo_ss.w, nr_wo_ss.r, nr_wo_ss.b, nr_wo_ss.W, nr_wo_ss.CV], EL_w_SS=[el_ss.K, el_ss.L, el_ss.w, el_ss.r, el_ss.b, el_ss.W, el_ss.CV], EL_wo_SS=[el_wo_ss.K, el_wo_ss.L, el_wo_ss.w, el_wo_ss.r, el_wo_ss.b, el_wo_ss.W, el_wo_ss.CV])

writedlm("data_summary.csv", Iterators.flatten(([names(data_summary)], eachrow(data_summary))), ',')
println("All done!")
