# ------------------------------------------------------------------------------
# Author: Heejin Yoon
# Problem Set 1 (Jean-Francois Houde)
# November 8, 2021
# ps1_model.jl
# ------------------------------------------------------------------------------

@with_kw struct Primitives
    data = DataFrame(load("Mortgage_performance_data.dta"))
    y::Array{Float64} = convert(Array{Float64}, select(data, :i_close_first_year))
    n::Int64 = length(y)
    X::Array{Float64} = [ones(n) convert(Array{Float64}, select(data, [:i_large_loan, :i_medium_loan, :rate_spread, :i_refinance, :age_r, :cltv, :dti, :cu, :first_mort_r, :score_0, :score_1, :i_FHA, :i_open_year2, :i_open_year3, :i_open_year4, :i_open_year5]))]
end

function loglikelihood(prim::Primitives, β::Array{Float64})
    @unpack X, y, n = prim

    Λ = zeros(n)
    l = 0.0

    Λ = (exp.(X * β) ./ (1 .+ exp.(X * β_initial)))
    l = sum(log.(Λ .^ (y) .* ((1 .- Λ) .^ (1 .- y))))

    l
end

function score(prim::Primitives, β::Array{Float64})
    @unpack X, y, n = prim

    Λ = zeros(n)
    g = zeros(size(X, 2))

    Λ = (exp.(X * β) ./ (1 .+ exp.(X * β_initial)))
    g = X' * (y - Λ)

    g
end

function hessian(prim::Primitives, β::Array{Float64})
    @unpack X, y, n = prim

    Λ = zeros(n)
    H = zeros(size(X, 2), size(X, 2))

    Λ = (exp.(X * β) ./ (1 .+ exp.(X * β_initial)))
    H = -(X' .* Λ') * ((1 .- Λ) .* X)

    H
end

function score_numerical(prim::Primitives, β::Array{Float64})
    @unpack X, y, n = prim

    ε = 1e-10
    g = zeros(size(X, 2))

    for i_index = 1:size(X, 2)
        β_up = β
        β_up[i_index] += ε
        β_down = β
        β_down[i_index] += (-ε)
        
        g[i_index] = (loglikelihood(prim, β_up) - loglikelihood(prim, β_down)) / (2 * ε)
    end

    g
end

function hessian_numerical(X::Array{Float64}, β::Array{Float64}, y::Array{Float64})
    ε = 1e-10
    H = zeros(length(X[1, :]), length(X[1, :]))
    for i = 1:length(X[1, :])

        β_upper = copy(β)
        β_upper[i] += ε
        g_upper = score_numerical(X, β_upper, y)

        β_lower = copy(β)
        β_lower[i] -= ε
        g_lower = score_numerical(X, β_lower, y)

        H[i, :] = (g_upper .- g_lower) ./ (2 * ε)
    end

    H
end

function Newton_method(X::Array{Float64}, β::Array{Float64}, y::Array{Float64})

    diff = 100
    tol = 10e-12
    n = 0
    β = zeros(length(X[1, :]))

    while diff > tol
        n += 1
        g = score(X, β, y)
        H = hessian(X, β, y)

        next_β = β .- inv(H) * g

        diff = maximum(abs.(next_β - β))
        β = next_β
        println("Iteration #", n, " / Difference: ", diff)
    end

    β
end
