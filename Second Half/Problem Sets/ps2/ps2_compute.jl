using Parameters, Plots, Optim, StatFiles, DataFrames, CSV, Random, .Threads

rt = pwd()

include(rt * "/Problem Sets/ps2/ps2_model.jl")

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


## Solve for MLE
optim_alt = optimize(theta -> -loglikelihood_quardrature_alt(prim, theta, X, Z, T)[2], res.θ, BFGS(), Optim.Options(show_trace=true, iterations=50))
optim = optimize(theta -> loglikelihood_quardrature_alt(prim, theta, X, Z, T)[2], res.θ, BFGS(), Optim.Options(show_trace=true, iterations=50))

θ_candidate_alt = optim_alt.minimizer
θ_candidate = optim.minimizer

