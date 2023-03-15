# ------------------------------------------------------------------------------
# Author: Heejin
# Hopenhayn and Rogerson (1993, JPE)
# October 21, 2021
# ps6_compute.jl
# ------------------------------------------------------------------------------

using Parameters, Interpolations, Optim, LinearAlgebra

# keyword-enabled structure to hold model Primitives

@with_kw struct Primitives
    cf::Float64 = 10.0
    α::Float64 = -1.0
    β::Float64 = 0.8                                                             # discount rate
    θ::Float64 = 0.64                                                            # coefficient of relative risk aversion
    A::Float64 = 1/200
    ce::Float64 = 5.0
    s_grid::Array{Float64} = [3.98e-4; 3.58; 6.82; 12.18; 18.79]
    n_s::Int64 = length(s_grid)
    F::Array{Float64} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                         0.1997 0.7201 0.0420 0.0326 0.0056;
                         0.2 0.2 0.5555 0.0344 0.0101;
                         0.2 0.2 0.2502 0.3397 0.0101;
                         0.2 0.2 0.25 0.34 0.01]
    nu::Array{Float64} = [0.37, 0.4631, 0.1102, 0.0504, 0.0063]
end

# structure that holds model results

mutable struct Results
    p::Float64
    N_d::Array{Float64}
    π::Array{Float64}
    exit_func::Array{Float64}
    val_func::Array{Float64}
    M::Float64
    μ::Array{Float64}
    Π::Float64
    L_d_incumbent::Float64
    L_d_entrant::Float64
    frac_L_d_entrant::Float64
    L_d::Float64
    L_s::Float64
    μ_incumbent::Float64
    μ_entrant::Float64
    μ_exit::Float64
end

# function for initializing model primitives and results

function Initialize()
    prim = Primitives()                                                          # initialize primtiives
    n_s = prim.n_s
    p = 1.0
    N_d = zeros(n_s)
    π = zeros(n_s)
    exit_func = ones(n_s)
    val_func = zeros(n_s)
    M = 5.0
    μ = ones(n_s)
    Π = 0.0
    L_d_incumbent = 0.0
    L_d_entrant = 0.0
    frac_L_d_entrant = 0.0
    L_d = 0.0
    L_s = 0.0
    μ_incumbent = 0.0
    μ_entrant = 0.0
    μ_exit = 0.0
    res = Results(p, N_d, π, exit_func, val_func, M, μ, Π, L_d_incumbent, L_d_entrant, frac_L_d_entrant, L_d, L_s, μ_incumbent, μ_entrant, μ_exit)
    prim, res                                                                    # return deliverables
end

function labor_demand(prim::Primitives, res::Results)
    @unpack θ, s_grid, n_s = prim
    @unpack p = res

    l = zeros(n_s)
    for s_index = 1:n_s
        l[s_index] = max((p*s_grid[s_index]*θ)^(1/(1-θ)), 0)
    end

    l
end

function static_profit(prim::Primitives, res::Results)
    @unpack cf, θ, s_grid, n_s = prim
    @unpack p, N_d = res

    profit = zeros(n_s)
    for s_index = 1:n_s
        profit[s_index] = p*s_grid[s_index]*N_d[s_index]^θ - N_d[s_index] - p * cf
    end

    profit
end

# Firm Problem

function Bellman(prim::Primitives, res::Results)
    @unpack n_s, β, F = prim                      # unpack model primitives
    @unpack π, val_func = res                                       # unpack results

    exit_func_next = zeros(n_s)
    val_func_next = zeros(n_s)                                             # next guess of value function to fill

    for s_index = 1:n_s
        w_exit = π[s_index]
        w_stay = π[s_index]
        for sp_index = 1:n_s
            w_stay += β*F[s_index, sp_index]*val_func[sp_index]
        end

        if w_stay>=w_exit
            exit_func_next[s_index] = 0.0
            val_func_next[s_index] = w_stay
        else
            exit_func_next[s_index] = 1.0
            val_func_next[s_index] = w_exit
        end
    end

    exit_func_next, val_func_next                                                                       # return next guess of value function
end

# Value function iteration

function V_iterate(prim::Primitives, res::Results)
    @unpack val_func, exit_func = res
    @unpack α = prim

    n = 0                                                                        # count
    tol = 0.0
    err = 100.0

    res.N_d = labor_demand(prim, res)
    res.π = static_profit(prim, res)

    if α==-1
        while err>0                                                               # begin iteration
            n+=1
            exit_func_next, val_func_next = Bellman(prim, res)                                            # spit out new vectors
            err = sum(abs.(res.val_func .- val_func_next)) + sum(abs.(res.exit_func .- exit_func_next))
            res.val_func = val_func_next
            res.exit_func = exit_func_next
            # println("number of iteration: ", n, ", error size: ", err)
        end
    else
        while err>0                                                               # begin iteration
            n+=1
            exit_func_next, val_func_next = Bellman_random(prim, res)                                            # spit out new vectors
            err = sum(abs.(res.val_func .- val_func_next)) + sum(abs.(res.exit_func .- exit_func_next))
            res.val_func = val_func_next
            res.exit_func = exit_func_next
            # println("number of iteration: ", n, ", error size: ", err)
        end
    end

    println("Firm problem converged after ", n, " iterations.")
end

# Price

function entry_condition(prim::Primitives, res::Results)
    @unpack nu, ce = prim
    @unpack val_func, p = res

    sum(val_func .* nu)/p - ce
end

function stationary_price(prim::Primitives, res::Results)
    n = 0                                                                        # count
    tol = 1e-4
    EC = 100.0
    p_max = 10.0
    p_min = 1e-4
    candidate_p = (p_min+p_max)/2

    while abs(EC)>tol
        n += 1
        V_iterate(prim, res)
        EC = entry_condition(prim, res)
        if EC < 0
            p_min = candidate_p
        elseif EC >= 0
            p_max = candidate_p
        end
        candidate_p = (p_min+p_max)/2
        res.p = candidate_p
    end

    println("Stationary price is found after ", n, " iterations.")
end

function Tstar_Bellman(prim::Primitives, res::Results)
    @unpack n_s, F, nu = prim
    @unpack exit_func, val_func, μ = res

    μ_next = zeros(n_s)
    for s_index = 1:n_s, sp_index = 1:n_s
        μ_next[sp_index] += (1 - exit_func[s_index]) * F[s_index, sp_index] * μ[s_index]
        μ_next[sp_index] += (1 - exit_func[s_index]) * F[s_index, sp_index] * res.M * nu[s_index]
    end

    μ_next
end

# Value function iteration

function μ_iterate(prim::Primitives, res::Results)
    @unpack n_s, F, nu = prim
    @unpack exit_func, μ = res

    n = 0                                                                        # count
    tol = 1e-4
    err = 100.0

    while err>tol
        n += 1
        μ_next = Tstar_Bellman(prim, res)
        err = maximum(abs.(μ_next.-res.μ))
        res.μ = μ_next
    end

    μ_incumbent = 0.0
    μ_entrant = 0.0
    μ_exit = 0.0
    μ_incum = zeros(n_s)
    μ_ent = zeros(n_s)
    μ_ex = zeros(n_s)
    for s_index = 1:n_s, sp_index = 1:n_s
        μ_incum[sp_index] += (1 - exit_func[s_index]) * F[s_index, sp_index] * μ[s_index]
        μ_ent[sp_index] += (1 - exit_func[s_index]) * F[s_index, sp_index] * res.M * nu[s_index]
        μ_ex[sp_index ] += exit_func[s_index] * F[s_index, sp_index] * μ[s_index] + exit_func[s_index] * F[s_index, sp_index] * res.M * nu[s_index]
    end
    res.μ_incumbent = sum(μ_incum)
    res.μ_entrant = sum(μ_ent)
    res.μ_exit = sum(μ_ex)
    println("Invariant μ converged in ", n, " iterations")
end

function compute_LMC(prim::Primitives, res::Results)
    @unpack A, nu = prim
    @unpack N_d, μ, M, π = res

    L_d_incumbent = sum(N_d .* μ)
    L_d_entrant = M * sum(N_d .* nu)
    L_d = L_d_incumbent + L_d_entrant
    frac_L_d_entrant = L_d_entrant/L_d
    Π = sum(π .* μ) + M * sum(π .* nu)
    L_s = 1/A - Π
    res.L_d_incumbent, res.L_d_entrant, res.frac_L_d_entrant, res.L_d, res.Π, res.L_s = L_d_incumbent, L_d_entrant, frac_L_d_entrant, L_d, Π, L_s
    LMC = L_d - L_s

    LMC
end

function stationary_M(prim::Primitives, res::Results)

    n = 0                                                                        # count
    tol = 0.005
    LMC = 100.0
    M_max = 10.0
    M_min = 1e-4
    candidate_M = (M_min+M_max)/2

    while abs(LMC)>tol
        n += 1
        μ_iterate(prim, res)
        LMC = compute_LMC(prim, res)

        if LMC > 0
            M_max = candidate_M
        else
            M_min = candidate_M
        end
        candidate_M = (M_min+M_max)/2
        res.M = candidate_M
        println(M_min, "/", M_max)
    end

    println("Stationary M is found after ", n, " iterations.")
end

# solve the model

function solve_model(prim::Primitives, res::Results)
    stationary_price(prim, res)
    stationary_M(prim, res)

    res
end

function Bellman_random(prim::Primitives, res::Results)
    @unpack n_s, β, F, α = prim                      # unpack model primitives
    @unpack π, val_func = res                                       # unpack results

    exit_func_next = zeros(n_s)
    val_func_next = zeros(n_s)                                             # next guess of value function to fill

    for s_index = 1:n_s
        w_exit = π[s_index]
        w_stay = π[s_index]
        for sp_index = 1:n_s
            w_stay += β*F[s_index, sp_index]*val_func[sp_index]
        end

        c = max(α * w_stay, α * w_exit)
        val_func_next[s_index] = MathConstants.eulergamma/α + 1/α*(c+log(exp(α*w_stay - c)+exp(α*w_exit - c)))
        exit_func_next[s_index] = exp(α * w_exit - c) / (exp.(α * w_stay - c) + exp(α * w_exit - c))
    end

    exit_func_next, val_func_next                                                                       # return next guess of value function
end
