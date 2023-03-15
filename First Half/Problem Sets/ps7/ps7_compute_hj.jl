# ------------------------------------------------------------------------------
# Author: Heejin
# Hopenhayn and Rogerson (1993, JPE)
# October 21, 2021
# ps6_model.jl
# ------------------------------------------------------------------------------
using Parameters, Random, Optim, Distributions, LinearAlgebra, StatsBase

include("ps7_model_hj.jl");                                                         # import the functions that solve our growth model

prim, est = Initialize()


res1 = SMM(prim, est, 1, 2)

res2 = SMM(prim, est, 2, 3)

res3 = SMM(prim, est, 1, 3)
