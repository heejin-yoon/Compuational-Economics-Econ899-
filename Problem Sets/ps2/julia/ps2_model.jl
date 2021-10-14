# ------------------------------------------------------------------------------
# Author: Heejin
# Huggett (1993, JEDC)
# September 20, 2021
# ps2_model.jl
# ------------------------------------------------------------------------------

using Parameters, Interpolations, Optim, LinearAlgebra

# keyword-enabled structure to hold model Primitives

@with_kw struct Primitives
    β::Float64 = 0.9932                                                          # discount rate
    α::Float64 = 1.5                                                             # coefficient of relative risk aversion
    S::Array{Float64, 1} = [1.0, 0.5]                                            # set of possible earning states
    n_S::Int64 = length(S)                                                       # number of earning states
    Π::Array{Float64, 2} = [[0.97, 0.5] [0.03, 0.5]]                             # transition matrix
    a_lb::Int64 = -2.0                                                           # lower asset boundary
    a_ub::Int64 = 5.0                                                            # upper asset boundary
    n_a::Int64 = 1000                                                            # number of asset grids
    a_grid::Array{Float64, 1} = collect(range(a_lb, stop = a_ub, length = n_a))  # savings grid array
end

# structure that holds model results

mutable struct Results
    val_func::Array{Float64, 2}                                                  # value function
    pol_func::Array{Float64, 2}                                                  # policy function
    μ::Array{Float64, 2}                                                         # bond holdings distribution
    q::Float64                                                                   # bond price
end

# function for initializing model primitives and results

function Initialize()
    prim = Primitives()                                                          # initialize primtiives
    val_func = [zeros(prim.n_a) zeros(prim.n_a)]                                 # initial val_func guess: zero
    pol_func = [zeros(prim.n_a) zeros(prim.n_a)]                                 # initial pol_func guess: zero
    μ        = [ones(prim.n_a) ones(prim.n_a)]/(prim.n_a*2)                      # uniform bond holding distribution
    q        = (prim.β+1)/2                                                      # initial q guess: the midpoint of β and 1
    res = Results(val_func, pol_func, μ, q)                                      # initialize results struct
    prim, res                                                                    # return deliverables
end

# HH Problem
# Bellman Operator

function T_Bellman(prim::Primitives, res::Results)
    @unpack val_func, pol_func, μ, q = res                                       # unpack results
    @unpack a_grid, a_lb, a_ub, n_a, S, n_S, Π, α, β = prim                      # unpack model primitives
    v_next = [zeros(n_a) zeros(n_a)]                                             # next guess of value function to fill

    for a_index = 1:n_a
        a = a_grid[a_index]                                                      # value of assets

        for S_index = 1:n_S
            s = S[S_index]
            candidate_max = -Inf                                                 # bad candidate max
            budget = s + a                                                       # budget

            for ap_index in 1:n_a                                                # loop over possible selections of k', exploiting monotonicity of policy function
                c = budget - a_grid[ap_index]*q                                  # consumption given k' selection

                if c>0                                                           # check for positivity
                    val = (c^(1-α)-1)/(1-α)                                      # compute value

                    for Sp_index in 1:n_S
                        val = val + β*Π[S_index,Sp_index]*val_func[ap_index, Sp_index]
                    end

                    if val>candidate_max                                         # check for new max value
                        candidate_max = val                                      # update max value
                        pol_func[a_index, S_index] = a_grid[ap_index]            # update policy function
                    end
                end
            end
            v_next[a_index, S_index] = candidate_max                             # update value function
        end
    end
    v_next                                                                       # return next guess of value function
end

# Value function iteration

function V_iterate(prim::Primitives, res::Results, tol::Float64 = 1e-4, err::Float64 = 100.0)

    n = 0                                                                        # count

    while err>tol                                                                # begin iteration
        n+=1

        v_next = T_Bellman(prim, res)                                            # spit out new vectors
        err = abs.(maximum(v_next.-res.val_func))/abs(maximum(v_next))           # reset error level
        res.val_func = v_next                                                    # update value function
    end

    println("HH problem converged in ", n, " iterations.")
end

# Bond holdings distribution
# Bellman Operator

function Tstar_Bellman(prim::Primitives, res::Results, progress::Bool = false)
    @unpack pol_func, μ = res
    @unpack n_a, n_S, Π = prim

    μ_next = zeros(n_a, n_S)

    for a_index = 1:n_a
        for S_index = 1:n_S

            #if progress
            #    println(i_s, ", ", i_a)
            #end
            ap = pol_func[a_index, S_index]

            ap_increase_e = argmin(abs.(ap .- a_grid))
            ap_increase_u = argmin(abs.(ap .- a_grid))

            μ_next[ap_increase_e, 1] = μ_next[ap_increase_e, 1] + μ[a_index, S_index] * Π[S_index, 1]
            μ_next[ap_increase_u, 2] = μ_next[ap_increase_u, 2] + μ[a_index, S_index] * Π[S_index, 2]

        end
    end
    μ_next
end

# Value function iteration

function μ_iterate(prim::Primitives, res::Results, tol::Float64 = 1e-4, err::Float64 = 100.0, progress::Bool = false)

    n = 0                                                                        # count

    while err>tol

        n += 1

        μ_next = Tstar_Bellman(prim, res)
        err = abs.(maximum(μ_next.-res.μ))/abs(maximum(μ_next))
        res.μ = μ_next
    end

    println("Invariant μ converged in ", n, " iterations")
end

#

function Update_price(prim::Primitives, res::Results, tol::Float64 = 1e-3)

    @unpack a_grid, β = prim

    excess_demand = -sum(res.μ.* [a_grid a_grid])

    if excess_demand > tol
        q_hat = res.q + (β - res.q)/2*abs(excess_demand)

        println("Excess Demand is positive: ", excess_demand)
        println("Lower bond price from ", res.q, " to ", q_hat)

        res.q = q_hat

        return(false)

    elseif excess_demand < -tol
        q_hat = res.q + (1 - res.q)/2*abs(excess_demand)

        println("Excess Demand is negative: ", excess_demand)
        println("Raise bond price from ", res.q, " to ", q_hat)

        res.q = q_hat

        return(false)

    else
        println("Excess Demand is within tolerence: ", excess_demand)

        return(true)
    end
end

# solve the model

function Solve_model(prim::Primitives, res::Results)

    converged = false

    while !converged
        V_iterate(prim, res)
        μ_iterate(prim, res)
        converged = Update_price(prim, res)
    end
end

# wealth distribution calculation

function Wealth_distribution(prim::Primitives, res::Results)
    @unpack a_grid, S, n_S, n_a = prim

    wealth = [zeros(n_a) zeros(n_a)]

    for a_index = 1:n_a
        for S_index = 1:n_S

            w_index = argmin(abs.(a_grid[a_index] .+ S[S_index] .- a_grid))

            wealth[w_index, S_index] = res.μ[a_index, S_index]
        end
    end
    wealth
end

# calculate Lorenz curve

function Lorenz_curve(prim::Primitives, wealth::Array{Float64, 2})
    @unpack a_grid, n_a = prim

    x = cumsum(wealth[:,1] .+ wealth[:,2])
    y = cumsum((wealth[:,1] .+ wealth[:,2]) .* a_grid)

    unique([x/x[n_a] y/y[n_a]]; dims = 1)
end

# calculate Gini coefficient

function Gini(lorenz::Array{Float64, 2})
    widths = diff(lorenz[:,1])
    heights = ((lorenz[1:end-1,1] .+ lorenz[2:end,1])./2 .- (lorenz[1:end-1,2] .+ lorenz[2:end,2])./2)
    area1 = sum(widths .* heights)

    l_pos = lorenz[lorenz[:,2].>0, :]
    widths = diff(l_pos[:,1])
    heights = (l_pos[1:end-1,2] .+ l_pos[2:end,2])./2
    area2 = sum(widths .* heights)

    area1/(area1+area2)
end

# welfare change

function Welfare_fb(prim::Primitives)
    @unpack α, β, S, Π = prim
    stationary_distribution = (Π^1000000)[1, :]
    c_fb = stationary_distribution[1] * S[1] + stationary_distribution[2] * S[2]
    ((c_fb)^(1 - α) - 1)/((1 - α) * (1 - β))
end

function calculate_λ(prim::Primitives, res::Results, welfare_fb::Float64)
    @unpack α, β = prim

    numerator = welfare_fb + 1 /((1 - α)*(1 - β))
    denominator = res.val_func .+ (1 ./((1 .- α).*(1 .- β)))

    (numerator./denominator).^(1/(1 .- α)) .- 1
end
