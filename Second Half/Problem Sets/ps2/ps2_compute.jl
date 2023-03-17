using Parameters, Plots, Optim, StatFiles, DataFrames, CSV, Random, Distributions, .Threads, FloatingTableView

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps2/ps2_model.jl")

prim, res = Initialize()

## Set X, Z, Y, T values

X = select(prim.data, :score_0, :rate_spread, :i_large_loan, :i_medium_loan, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5)
Y = select(prim.data, :i_close_0, :i_close_1, :i_close_2)
Y[:, :T] .= 1.0
Y[(Y.i_close_0.==0.0).&(Y.i_close_1.==1.0), :T] .= 2.0
Y[(Y.i_close_0.==0.0).&(Y.i_close_1.==0.0).&(Y.i_close_2.==1.0), :T] .= 3.0
Y[(Y.i_close_0.==0.0).&(Y.i_close_1.==0.0).&(Y.i_close_2.==0.0), :T] .= 4.0
T = select(Y, :T)
Z = select(prim.data, :score_0, :score_1, :score_2)
X = Array{Float64}(X)
Z = Array{Float64}(Z)
T = Array{Float64}(T)

# res.θ = [0.0061751972825785624
#     1.220060239340399
#     -17.008615945472318
#     0.0018117495054437106
#     -0.000968136362492816
#     -0.001580543507744522
#     -0.0011558920192594763
#     -0.00013071142705024796
#     -5.2621090951833314e-5
#     0.0001990199344835649
#     -0.00020628735438775667
#     -0.0016216370604782437
#     -0.0010281241504542808
#     -0.0010298735485978336
#     -0.0021931798123920295
#     -0.00040926267287037495
#     0.0006985996606707934
#     0.0005650677232522111
#     -0.0011704093057336884
#     -13.978456563161936]

## Solve for MLE

optim = optimize(theta -> -loglikelihood_quardrature(prim, theta, X, Z, T)[2], res.θ, BFGS(), Optim.Options(show_trace=true, iterations=200))


