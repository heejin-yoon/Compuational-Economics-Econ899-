## Open Packages and Files

using Interpolations, Plots, Parameters, DataFrames, GLM
include("ps5_model_hj.jl");                                                         # import the functions that solve our growth model
prim, res = Initialize()
@elapsed solve_model(prim, res)
