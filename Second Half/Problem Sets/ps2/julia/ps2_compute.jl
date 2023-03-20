using Parameters, Plots, Optim, StatFiles, DataFrames, CSV, Random, Distributions, .Threads, FloatingTableView, StatsBase

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps2/julia/ps2_model.jl")

prim, res = Initialize()


## Set-up X, Z, Y, T values

X = select(prim.data, :i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5)
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


## Random draw from uniform distribution through halton sequence

n_draw = 100
u_0 = halton(3, n_draw)
u_1 = halton(5, n_draw)
u_2 = halton(7, n_draw)
# Random.seed!(123)
# u_0 = rand(Uniform(0, 1), n_trials)
# Random.seed!(1234)
# u_1 = rand(Uniform(0, 1), n_trials)


## Solve for MLE

# Quadrature

optim_quadrature = optimize(theta -> -loglikelihood_quardrature(prim, theta, X, Z, T)[2], prim.θ_initial_quadrature, BFGS(), Optim.Options(show_trace=true, iterations=10))
res.θ_quadrature = optim_quadrature.minimizer

# GHK

optim_ghk = optimize(theta -> -loglikelihood_ghk(prim, theta, X, Z, T, u_0, u_1)[2], prim.θ_initial_ghk, BFGS(), Optim.Options(show_trace=true, iterations=10))
res.θ_ghk = optim_ghk.minimizer

# Accept/Reject Method

optim_acceptreject = optimize(theta -> -loglikelihood_acceptreject(prim, theta, X, Z, T, u_0, u_1, u_2)[2], prim.θ_initial_acceptreject, BFGS(), Optim.Options(show_trace=true, iterations=10))
res.θ_acceptreject = optim_acceptreject.minimizer

params = DataFrame([["α₀"; "α₁"; "α₂"; "β_i_large_loan"; "β_i_medium_loan"; "β_rate_spread"; "β_i_refinance"; "β_age_r"; "β_cltv"; "β_dti"; "β_cu"; "β_first_mort_r"; "β_score_0"; "β_i_FHA"; "β_i_open_year2"; "β_i_open_year3"; "β_i_open_year4"; "β_i_open_year5"; "γ"; "ρ"] res.θ_quadrature res.θ_ghk res.θ_acceptreject], [:Parameters, :Quadrature, :GHK, :AcceptReject])

predicted_probs = DataFrame(Stats=["Mean", "Max", "P75", "Median", "P25", "Min"], Quadrature=[mean(res.L_quadrature), maximum(res.L_quadrature), percentile(res.L_quadrature, 75), median(res.L_quadrature), percentile(res.L_quadrature, 25), minimum(res.L_quadrature)], GHK=[mean(res.L_ghk), maximum(res.L_ghk), percentile(res.L_ghk, 75), median(res.L_ghk), percentile(res.L_ghk, 25), minimum(res.L_ghk)], AcceptReject=[mean(res.L_acceptreject), maximum(res.L_acceptreject), percentile(res.L_acceptreject, 75), median(res.L_acceptreject), percentile(res.L_acceptreject, 25), minimum(res.L_acceptreject)])