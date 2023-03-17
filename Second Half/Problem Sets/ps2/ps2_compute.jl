using Parameters, Plots, Optim, StatFiles, DataFrames, CSV, Random, Distributions, .Threads, FloatingTableView

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps2/ps2_model.jl")

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

optim_quadrature = optimize(theta -> -loglikelihood_quardrature(prim, theta, X, Z, T)[2], prim.θ_initial, BFGS(), Optim.Options(show_trace=true, iterations=200))
res.θ_quadrature = optim_quadrature.minimizer

# GHK

optim_ghk = optimize(theta -> -loglikelihood_ghk(prim, theta, X, Z, T, u_0, u_1)[2], prim.θ_initial, BFGS(), Optim.Options(show_trace=true, iterations=200))
res.θ_ghk = optim_ghk.minimizer

# Accept/Reject Method

optim_acceptreject = optimize(theta -> -loglikelihood_acceptreject(prim, theta, X, Z, T, u_0, u_1, u_2)[2], prim.θ_initial, BFGS(), Optim.Options(show_trace=true, iterations=200))
res.θ_acceptreject = optim_acceptreject.minimizer

a, b = loglikelihood_ghk(prim, prim.θ_initial, X, Z, T, u_0, u_1)

mean(a)
median(a)
maximum(a)
minimum(a)

a, b = loglikelihood_acceptreject(prim, prim.θ_initial, X, Z, T, u_0, u_1, u_2)

mean(a)
median(a)
maximum(a)
minimum(a)