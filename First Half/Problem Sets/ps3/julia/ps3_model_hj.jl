# ------------------------------------------------------------------------------
# Author: Heejin
# Conesa and Krueger (1999, RED)
# September 29, 2021
# ps3_model.jl
# ------------------------------------------------------------------------------

using Parameters

# keyword-enabled structure to hold model Primitives

@with_kw struct Primitives                                                       # define parameters
    N::Int64 = 66                                                                # lifespan
    n::Float64 = 0.011                                                           # population growth rate
    J_retire::Int64 = 46                                                         # retirement age
    θ::Float64 = 0.11                                                            # labor income tax rate
    σ::Float64 = 2.0                                                             # relative risk aversion coefficient
    γ::Float64 = 0.42                                                            # weight on consumption in utility function
    η::Array{Float64, 1} = [0.59923239, 0.63885106, 0.67846973, 0.71808840, 0.75699959, 0.79591079, 0.83482198, 0.87373318, 0.91264437, 0.95155556, 0.99046676, 0.99872065,  1.0069745,  1.0152284,  1.0234823,  1.0317362,  1.0399901,  1.0482440,  1.0564979,  1.0647518,  1.0730057,  1.0787834,  1.0845611,  1.0903388,  1.0961165,  1.1018943,  1.1076720,  1.1134497,  1.1192274, 1.1250052, 1.1307829, 1.1233544, 1.1159259, 1.1084974, 1.1010689, 1.0936404, 1.0862119, 1.0787834, 1.0713549,  1.0639264, 1.0519200, 1.0430000, 1.0363000, 1.0200000, 1.0110000]                       # deterministic age-efficiency
    z::Array{Float64, 1} = [3.0, 0.5]                                            # idiosyncratic productivity
    e::Matrix{Float64} = η*z'                                                    # worker's productivity (z*η: 45x2 matrix)
    n_z::Int64 = length(z)                                                       # number of productivity states
    Π₀::Array{Float64, 1} = [0.2037, 0.7963]                                     # idiosyncratic productivity
    Π::Matrix{Float64} = [0.9261 1-0.9261;1-0.9811 0.9811]                       # persistence probabilities [πₕₕ πₕₗ; πₗₕ πₗₗ]
    α::Float64 = 0.36                                                            # capital share
    δ::Float64 = 0.06                                                            # depreciation rate
    β::Float64 = 0.97                                                            # discount rate
    a_lb::Int64 = 0.0                                                            # lower asset boundary
    a_ub::Int64 = 75.0                                                           # upper asset boundary
    n_a::Int64 = 1000                                                            # number of asset grids
    a_grid::Array{Float64, 1} = collect(range(a_lb, stop = a_ub, length = n_a))  # asset grid array
end

# structure that holds model results

mutable struct Results
    val_func::Array{Float64}                                                     # value function
    pol_func::Array{Float64}                                                     # policy function
    lab_func::Array{Float64}                                                     # labor function
    K0::Float64                                                                  # aggregate capital
    L0::Float64                                                                  # aggregate labor
    μ::Array{Float64}                                                            # asset distribution
    w::Float64                                                                   # wage
    r::Float64                                                                   # interest rate
    b::Float64                                                                   # pension benefit
    W::Float64                                                                   # welfare
    CV::Float64                                                                  # CV
end

# function for initializing model primitives and results

function Initialize()                                                            # initialize primtives
    @unpack N, n_a, n_z, J_retire, z = prim
    val_func = zeros(N, n_a, n_z)                                                # initial val_func guess: zero
    pol_func = zeros(N, n_a, n_z)                                                # initial pol_func guess: zero
    lab_func = zeros(N, n_a, n_z)                                                # initial lab_func guess: zero
    K0 = 3.3                                                                     # initial guess on aggregate capital
    L0 = 0.3                                                                     # initial guess on aggregate labor
    μ = zeros(N, n_a, n_z)                                                       # initialize asset distribution
    w = 1.05                                                                     # initial assumption on w
    r = 0.05                                                                     # initial assumption on r
    b = 0.2                                                                      # initial assumption on b
    W = 0.0
    CV = 0.0
    res = Results(val_func, pol_func, lab_func, K0, L0, μ, w, r, b, W, CV)      # initialize results struct
end

## HH decision

# utility function of retired people

function u_retired(prim::Primitives, res::Results, c::Float64)
    @unpack σ, γ = prim

    if c>0
        u_r = (c^((1-σ) * γ))/(1 - σ)
    else
        u_r = -Inf
    end

    u_r
end

# Bellman equation for retired people

function Bellman_retired(prim::Primitives, res::Results)
    @unpack N, J_retire, a_grid, n_a, β = prim

    for a_index=1:n_a
        c = (1+res.r)*a_grid[a_index]+res.b
        res.val_func[N, a_index, 1] = u_retired(prim, res, c)
        # res.pol_func[N, a_index, 1] = 0
    end

    for j = N-1:-1:J_retire
        for a_index=1:n_a
            wealth = (1+res.r)*a_grid[a_index]+res.b
            candidate_max = -Inf
            for ap_index=1:n_a
                c = wealth-a_grid[ap_index]
                val = u_retired(prim, res, c) + β*res.val_func[j+1,ap_index,1]
                if val>candidate_max
                    candidate_max = val
                    res.pol_func[j, a_index, 1] = a_grid[ap_index]
                    #lab_func[j, a_index, 1] = 0
                end
            end
            res.val_func[j, a_index, 1] = candidate_max
        end
    end
    res.val_func[:, :, 2] = res.val_func[:, :, 1]
    res.pol_func[:, :, 2] = res.pol_func[:, :, 1]
    #lab_func[:, :, 2] = lab_func[:, :, 1]
end

# optimal labor choice

function optimal_labor(prim::Primitives, res::Results, a::Float64, ap::Float64, j::Int64, z_index::Int64)
    @unpack γ, θ, e, η = prim

    l::Float64 = (γ*(1-θ)*e[j,z_index]*res.w - (1-γ)*((1+res.r)*a - ap)) / ((1-θ)*res.w*e[j,z_index])
    if l>1
        l = 1
    elseif l<0
        l = 0
    end
    l
end

# utility function of workers

function u_worker(prim::Primitives, res::Results, c::Float64, l::Float64)
    @unpack σ, γ = prim

    if (c>0)&&(l>=0)&&(l<=1)
        u_w = (((c^γ) * ((1 - l)^(1-γ)))^(1-σ))/(1 - σ)
    else
        u_w = -Inf
    end

    u_w
end

# Bellman equation for workers

function Bellman_worker(prim::Primitives, res::Results)
    @unpack N, J_retire, a_grid, n_a, n_z, β, e, Π, θ = prim

    for j = (J_retire-1):-1:1
        for z_index=1:n_z, a_index=1:n_a
            candidate_max = -Inf
            for ap_index=1:n_a
                l = optimal_labor(prim, res, a_grid[a_index], a_grid[ap_index], j, z_index)
                c = res.w*(1-θ)*e[j,z_index]*l + (1+res.r)*a_grid[a_index] - a_grid[ap_index]
                val = u_worker(prim, res, c, l)
                for zp_index=1:n_z
                    val += β*Π[z_index, zp_index]*res.val_func[j+1,ap_index,zp_index]
                end

                if val>candidate_max
                    candidate_max = val
                    res.pol_func[j, a_index, z_index] = a_grid[ap_index]
                    res.lab_func[j, a_index, z_index] = l
                end
            end
            res.val_func[j, a_index, z_index] = candidate_max
        end
    end
end

# solve HH problem

function solve_HH_problem(prim::Primitives, res::Results)
    Bellman_retired(prim, res)
    Bellman_worker(prim, res)
end

## Steady-state distribution

# distribution of population according to the age and asset grid

function μ_distribution(prim::Primitives, res::Results)
    @unpack a_grid, N, n, n_a, n_z, Π₀, Π = prim

    res.μ[:, :, :] = zeros(N, n_a, n_z)                                          # initialize again
    res.μ[1, 1, :] = Π₀                                                          # j=1, a=0, z=zᴴ or zᴸ

    for j = 1:(N-1)
        for a_index=1:n_a, z_index=1:n_z
            if res.μ[j, a_index, z_index] == 0                                   # μ cannot evolve anymore
                continue
            end
            ap_choice = res.pol_func[j, a_index, z_index]
            for ap_index=1:n_a
                if a_grid[ap_index] == ap_choice
                    for zp_index=1:n_z
                        res.μ[j+1, ap_index, zp_index] += Π[z_index, zp_index]*res.μ[j, a_index, z_index]
                    end
                end
            end
        end
    end

    weight = ones(N)
    for i = 1:(N-1)
        weight[i+1,:,:] = weight[i,:,:]/(1+n)
    end
    sum_weight = sum(weight)
    age_weight = weight./sum_weight
    for z_index=1:n_z
        res.μ[:,:,z_index]= res.μ[:,:,z_index].*age_weight
    end
end

# aggregate economy-wide labor

function aggregate_L(prim::Primitives, res::Results)
    @unpack n_a, n_z, N, e, a_grid = prim

    E = zeros(N, n_a, n_z)
    for a_index=1:n_a, z_index=1:n_z
        E[1:45, a_index, z_index] = e[:,z_index]
    end
    L1 = sum(res.μ.*res.lab_func.*E)

    L1
end

# aggregate economy-wide capital

function aggregate_K(prim::Primitives, res::Results)
    @unpack n_a, n_z, N, e, a_grid = prim

    A_grid = zeros(N, n_a, n_z)
    for j=1:N, z_index=1:n_z
        A_grid[j, :, z_index] = a_grid
    end
    K1 = sum(res.μ.*A_grid)

    K1
end

# update prices

function update_prices(prim::Primitives, res::Results)
    @unpack α, δ, J_retire, N, θ = prim

    res.w = (1 - α) * (res.K0 ^ α) * (res.L0 ^ (- α))
    res.r = α * (res.K0 ^ (α - 1)) * (res.L0 ^ (1-α)) - δ
    res.b = (θ * res.w * res.L0) / sum(res.μ[J_retire:N, :, :])
end

## Solve the Model

function solve_model(prim::Primitives, res::Results)
    update_prices(prim, res)

    tol = 0.03
    i = 0
    λ = 0.5
    K1 = 0.0
    L1 = 0.0

    while true
        i += 1
        println("***********************************")
        println("Iteration #", i)

        solve_HH_problem(prim, res)
        println("solve_HH_problem is done.")

        μ_distribution(prim, res)
        println("μ_distribution is done.")

        K1 = aggregate_K(prim, res)
        L1 = aggregate_L(prim, res)
        println("aggregate_LK is done.")

        println("Capital demand: ", res.K0)
        println("Labor demand: ", res.L0)
        println("Capital supply: ", K1)
        println("Labor supply: ", L1)

        diff = abs(res.K0-K1)+abs(res.L0-L1)
        println("difference: ", diff)

        if diff > tol
            res.K0 = λ * K1 + (1 - λ) * res.K0
            res.L0 = λ * L1 + (1 - λ) * res.L0
            update_prices(prim, res)
        else
            break
        end
    end
    println("***********************************")
    println("★ Market Clearing Outcome ★  ")
    println(" ")

    println("Number of iterations:", i)
    println("Aggregate Capital:", K1)
    println("Aggregate Labor:", L1)
    println("Wage:", res.w)
    println("Interest rate:", res.r)
    println("Social security benefits:", res.b)
end

## Welfare Calculation

function welfare_analysis(prim::Primitives, res::Results)
    @unpack a_grid, n_a, n_z, N = Primitives()

    A_grid = zeros(N, n_a, n_z)
    welfare = res.val_func .* res.μ
    res.W = sum(welfare[isfinite.(welfare)])


    for j=1:N, z_index=1:n_z
        A_grid[j, :, z_index] = a_grid
    end

    wealth_first_moment = sum(res.μ.* A_grid)
    wealth_second_moment = sum(res.μ.* (A_grid.^2))
    wealth_variance = wealth_second_moment - (wealth_first_moment^2)
    res.CV = wealth_first_moment / sqrt(wealth_variance)
    println("Total welfare: ", res.W)
    println("CV: ", res.CV)

    res
end
