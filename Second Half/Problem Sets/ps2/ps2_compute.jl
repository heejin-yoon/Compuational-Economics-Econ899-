using Parameters, Plots, Optim, StatFiles, DataFrames, CSV, Random, Distributions, .Threads, FloatingTableView

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps2/ps2_model.jl")

prim, res = Initialize()


## Set X, Z, Y, T values

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

res.θ = [-5.48100226949492
    -2.614259560073641
    -2.2323474393428664
    0.40452302167241466
    0.27924492325612493
    0.264782325756695
    0.06258457636401359
    0.15085958657513318
    -0.04698336957419711
    0.10285115237450823
    0.4268824649599777
    0.21712408213320744
    -0.18340344234877518
    0.30116878763758176
    0.5115433213163416
    0.1339203500571433
    -0.0703953500654598
    -0.07471452242530689
    0.08134580158999291
    0.29460879975537024]


## Solve for MLE

optim = optimize(theta -> -loglikelihood_quardrature(prim, theta, X, Z, T)[2], res.θ, BFGS(), Optim.Options(show_trace=true, iterations=200))
res.θ = optim.minimizer
