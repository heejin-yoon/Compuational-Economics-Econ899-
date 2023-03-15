##
using Parameters, Plots, Random, Optim, Distributions, LinearAlgebra

cd("C:/Users/hyoon76/OneDrive - UW-Madison/5.Miscellaneous/CompEcon Practice/ps7/julia")
include("ps7_model_hj.jl");                                                         # import the functions that solve our growth model

##
prim, est = Initialize()

est.x, est.m = Truedata(prim)

plot(collect(1:prim.T), est.x, labels = "", title = "True data process")
savefig("truedata.png")

##

Identity = Matrix{Float64}(I, 3, 3)
est.b_hat1 = est_first(prim, est, Identity)

S = NeweyWest(prim, est)
W = S^(-1)
est.b_hat2 = est_first(prim, est, W)

##
res1 = SMM(prim, est, 1, 2)

res2 = SMM(prim, est, 2, 3)

res3 = SMM(prim, est, 1, 3)
