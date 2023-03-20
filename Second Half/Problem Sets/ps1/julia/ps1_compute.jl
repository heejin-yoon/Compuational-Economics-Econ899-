# ------------------------------------------------------------------------------
# Author: Heejin Yoon
# Problem Set 1 (Jean-Francois Houde)
# November 8, 2021
# ps1_compute.jl
# ------------------------------------------------------------------------------

using Parameters, StatFiles, DataFrames, Optim, Weave, FloatingTableView

rt = pwd()

include(rt * "/Second Half/Problem Sets/ps1/julia/ps1_model.jl")

prim = Primitives()
## Data Import

data = DataFrame(load("Mortgage_performance_data.dta"))
data[!, :constant] .= 1.0
y = select(data, :i_close_first_year)
X = select(data, [:constant, :i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :score_1, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5])
y = Float64.(Array(y))
y = convert(Array{Float64}, y)
X = convert(Array{Float64}, X)
β₀ = [-1.0]
β_initial = [β₀; zeros(size(X, 2) - 1)]

## Exercise 1.

L1 = loglikelihood(prim, β_initial)
g1 = score(prim, β_initial)
H1 = hessian(prim, β_initial)

# Calculations of log-likelyhood estimator, Score, and Hessian are done in L1, g1, and H1, respectively.

## Exercise 2.

g2 = score_numerical(X, β_initial, y)
H2 = hessian_numerical(X, β_initial, y)

# Numerical calculations of score and Hessian are done in g2 and H2, respectively.

## Exercise 3.

speed_newton = @elapsed β_newton = Newton_method(X, β_initial, y)
L_newton = loglikelihood(X, β_newton, y)

# β_newton contains the estimated coefficients using a Newton algorithm.

## Exercise 4.

speed_bfgs = @elapsed res_bfgs = optimize(β -> -loglikelihood(X, β, y), β_initial, BFGS())
β_bfgs = res_bfgs.minimizer
speed_simplex = @elapsed res_simplex = optimize(β -> -loglikelihood(X, β, y), β_newton, NelderMead());
β_simplex = res_simplex.minimizer;

# Calculations of estimates using BFGS and simplex method are done in β_bfgs and β_simplex.
# Numerical speed is calculated in speed_newton and speed_simplex, respectively.
