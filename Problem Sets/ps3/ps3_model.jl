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
    e = η*z'                                                                     # worker's productivity (z*η: 45x2 matrix)
    z::Array{Float64, 1} = [3.0, 0.5]                                            # idiosyncratic productivity
    n_z::Int64 = length(z                                                        # number of productivity states
    Π₀::Array{Float64, 1} = [0.2037, 0.7963]                                     # ergodic distribution of productivity
    Π::Matrix{Float64} = [0.9261 1-0.9261;1-0.9811 0.9811]                       # persistence probabilities [HH HL; LH LL]
    α::Float64 = 0.36                                                            # capital share
    δ::Float64 = 0.06                                                            # depreciation rate
    β::Float64 = 0.97                                                            # discount rate
    a_lb::Int64 = 0.0                                                            # lower asset boundary
    a_ub::Int64 = 75.0                                                           # upper asset boundary (why?)
    n_a::Int64 = 5000                                                            # number of asset grids (why?)
    a_grid::Array{Float64, 1} = collect(range(a_lb, stop = a_ub, length = n_a))  # asset grid array
end

# structure that holds model results

mutable struct Results
    val_func::Array{Float64}                                                     # value function
    pol_func::Array{Float64}                                                     # policy function
    lab_func::Array{Float64}                                                     # labor function
    μ::Array{Float64}                                                            # asset distribution
    L0::Float64                                                                  # aggregate labor
    K0::Float64                                                                  # aggregate capital
    w::Float64                                                                   # wage
    r::Float64                                                                   # interest rate
    b::Float64                                                                   # pension benefit
end

# function for initializing model primitives and results

function Initialize(prim::Primitives)                                            # initialize primtives
    @unpack N, n_a, n_z, J_retire = prim
    val_func = zeros(N, n_a, n_z)                                                # initial val_func guess: zero
    pol_func = zeros(N, n_a, n_z)                                                # initial pol_func guess: zero
    lab_func = zeros(N, n_a, n_z)                                                # initial lab_func guess: zero
    μ = zeros(N, n_a, n_z)                                                       # initial asset distribution guess (zero)
    K0 = 3.3
    L0 = 0.3
    w = 1.05                                                                     # initial assumption on w
    r = 0.05                                                                     # initial assumption on r
    b = 0.2                                                                      # initial assumption on b
    res = Results(val_func, pol_func, lab_func, μ, K0, L0, w, r, b)              # initialize results struct
    res                                                                          # return deliverables
end

############## HH decision ##############

# static labor supply decision

function optimal_labor(prim::Primitives, res::Results, a::Float64, ap::Float64, j::Int64, z_index::Int64)
    @unpack γ, θ, e, η = prim
    @unpack w, r = res

    l = (γ*(1-θ)*η[j,z_index]*w - (1-γ)*((1+r)*a - ap)) / ((1-θ)*w*η[j,z_index])

    if l>1
        l = 1
    elseif l<0
        l = 0
    end

    l
end

# utility grid of retired

function u_retired(prim::Primitives, res::Results)
    @unpack n_a, a_grid = prim
    @unpack r, b = res

    u_r = zeros(n_a,n_a,n_z)

    for a_index=1:n_a, ap_index=1:n_a, z_index=1:n_z
        a, ap = a_grid[a_index], a_grid[ap_index]
        c = (1+r)*a+b-ap

        if c>0
            u_r[a_index, ap_index, z_index] = (c^(γ*(1-σ)))/(1-σ)
        else
            u_r[a_index, ap_index, z_index] = -Inf
        end
    end
    u_r
end

# utility grid of worker

function u_worker(prim::Primitives, res::Results)
    @unpack n_a, a_grid = prim
    @unpack r, b = res

    u_w = zeros(j, n_a, n_a, n_z)

    for j=1:J_retire-1, a_index=1:n_a, ap_index=1:n_a, z_index=1:n_z
        a, ap, z = a_grid[a_index], a_grid[ap_index], z[z_index]
        l = optimal_labor(prim, res, a, ap, j, z)
        c = w*(1-θ)*η[j, z]*l+(1+r)*a-ap

        if c>0
            u_w[j, a_index, ap_index, z_index] = (c^γ*(1-l)^(1-γ))^(1-σ)/(1-σ)
        else
            u_w[j, a_index, ap_index, z_index] = -Inf
        end
    end
    u_w
end

'function update_price(prim::Primitives, res::Results)
    @unpack α, δ, J_retire, N = prim
    @unpack w, r, b, MU, θ, K0, L0 = res

    w = (1 - α) * K0 ^ α * L0 ^ (- α)
    r = α * K0 ^ (α - 1) * L0 ^ (1-α) - δ
    b = (θ * w * L0) / sum(MU[J_retire:N, :, :])
end'

# HH Problem for the retired
# Bellman Operator


function Bellman_retired(prim::Primitives, res::Results)
    @unpack θ, z, e, γ, val_func, pol_func, lab_func, MU, K0, L0, w, r, b = res   # unpack results
    @unpack N, J_retire, a_grid, n_a, n_z, Π, α, β, σ = prim                      # unpack model primitives
    val_new = zeros(n_a,n_z)

    for a_index=1:n_a, z_index = 1:n_z
        candidate_max = -Inf

        for j = N:-1:J_retire
            if j==N
                candidate_max = u_r[a_index,1]                                   # ap_index=1 => a'=0
                # pol_func[j, a_index, z_index] = 0
                val_func[j, a_index, z_index] = candidate_max
            else
                for ap_index=1:n_a
                    val = u_r[a_index,ap_index]+β*val_func[ap_index, z_index, j+1]

                    if val>candidate_max
                        candidate_max = val
                        pol_func[j, a_index, z_index] = a_grid[ap_index]
                        # lab_func[j, a_index, z_index] = 0
                    end
                end
                val_func[j, a_index, z_index] = candidate_max
            end
        end
    end
end

function Bellman_worker(prim::Primitives, res::Results)
    @unpack θ, z, e, γ, val_func, pol_func, lab_func, MU, K0, L0, w, r, b = res   # unpack results
    @unpack N, J_retire, a_grid, n_a, n_z, Π, α, β, σ = prim                      # unpack model primitives
    val_new = zeros(n_a,n_z)

    for a_index=1:n_a, z_index = 1:n_z
        candidate_max = -Inf

        for j = J_retire-1:-1:1
            for ap_index=1:n_a
                for zp_index=1:n_z
                    val = u_w[a_index+n_a*(z_index-1), ap_index, j]+β*Π[z_index,zp_index]*val_func[ap_index, z_index, j+1]
                    if val>candidate_max
                        candidate_max = val
                        a, ap, z = a_grid[a_index], ap_grid[ap_index], z[z_index]
                        pol_func[j, a_index, z_index] = a_grid[ap_index]
                        lab_func[j, a_index, z_index] = optimal_labor(prim, res, a, ap, j, z)
                    end
                end
            end
            val_func[j, a_index, z_index] = candidate_max
        end
    end
end

#computation of stationary distribution
function stationary_asset(prim::Primitives,res::Results)
    @unpack J, markov, zgrid, μ_0, N, Z, pop_growth, agrid = prim
    @unpack pol_func, μ = res

    μ[1,1,1] = Π₀[1]
    μ[1,1,2] = Π₀[2]

    for j = 1:(N-1)                                                              #loop over age
        for a_index = 1:n_a, z_index = 1:n_z                                     #loop over state space
            ap_choice = pol_func[j, a_index, z_index]                            #choice of a'
            for ap_index = 1:n_a, zp_index = 1:n_z                               #loop over tomorrow's states
                if a_grid[ap_index] == ap_choice                                 # check for consistency
                    stat_dist[index_ap,index_zp,j+1] += stat_dist[index_a,index_z,j]*markov[index_z,index_zp]*(1/(1+pop_growth)) #update next period's stationary dist
                end
            end
        end
    end
    res.stat_dist = stat_dist #update stat_dist object
end


function u_worker(c::Float64, l::Float64, γ::Float64, σ::Float64)
    if (c > 0 && l >= 0 && l <= 1)
        (((c^γ) * ((1 - l)^(1-γ)))^(1-σ))/(1 - σ)
    else
        -Inf
    end
end

# Solves workers problem. Need to solve retiree problem first.
function Bellman_worker(prim::Primitives, res::Results)
    @unpack θ, z, e, γ, val_func, pol_func, lab_func, MU, K0, L0, w, r, b = res             # unpack results
    @unpack a_grid, a_lb, a_ub, n_a, n_z, Π, α, β, σ, J_retire = prim                      # unpack model primitives

    # Backward induction starting period before retirement.
    for j = (J_retire-1):-1:1
        for z_index = 1:n_z
            choice_lower = 1 # exploits monotonicity of policy function
            for a_index = 1:n_a
                val_previous = -Inf
                for ap_index = choice_lower:n_a
                    l = labor_function(a_grid[a_index], a_grid[ap_index], e[j, z_index], θ, γ, r, w)
                    budget=w*(1-θ)*e[j,z_index]*l + (1+r)*a_grid[a_index]
                    val = u_worker(budget-a_grid[ap_index], l, γ, σ) # Instanteous utility
                    # iterates over tomorrow productivity to add continuation value
                    for zp_index = 1:n_z
                        val = val + β*Π[z_index, zp_index]*val_func[j+1, ap_index, zp_index]
                    end
                    if val<val_previous                                         # check for new max value
                        val_func[j, a_index, z_index] = val_previous                # update value function for age j
                        pol_func[j, a_index, z_index] = a_grid[ap_index-1]             # update policy function for age j
                        lab_func[j, a_index, z_index] = labor_function(a_grid[a_index],
                                a_grid[ap_index-1], e[j, z_index], θ, γ, r, w)
                        choice_lower = ap_index -1
                        break
                    elseif ap_index == n_a
                        val_func[j, a_index, z_index] = val                # update value function for age j
                        pol_func[j, a_index, z_index] = a_grid[ap_index]             # update policy function for age j
                        lab_func[j, a_index, z_index] = labor_function(a_grid[a_index],
                                a_grid[ap_index], e[j, z_index], θ, γ, r, w)
                    end
                    val_previous = val
                end
            end
        end
    end
end


# solves HH problem. Retiree first, then worker.
function solve_HH_problem(prim::Primitives, res::Results)
    Bellman_retired(prim, res)
    Bellman_worker(prim, res)
end

# aggregate asset distribution

function asset_dist(prim::Primitives, res::Results)
    @unpack N, n, n_z, Π₀, Π, n_a, a_grid = prim
    @unpack val_func, pol_func, lab_func, w, r, b = res                       # unpack results
    res.MU[1, 1, :] = Π₀                                                             # initial distribution: 0.2037 vs. 0.7963

    for j = 1:(N-1)
        for a_index = 1:n_a
            for z_index = 1:n_z
                if res.MU[j, a_index, z_index] == 0
                    continue
                end
                ap_index = argmax(a_grid .== pol_func[j, a_index, z_index])
                for zp_index = 1:n_z
                    res.MU[j+1, ap_index, zp_index] += Π[z_index, zp_index] * res.MU[j, a_index, z_index]
                end
            end
        end
    end
    total_pop = 1
    for j = 2:N
        res.MU[j,:,:] = res.MU[j,:,:]./(1+n)^(j-1)
        total_pop += 1/(1+n)^(j-1)
    end
    res.MU = res.MU./total_pop
end


function aggregate_lk(prim::Primitives, res::Results)
    @unpack J_retire, n_a, n_z, a_grid, N = prim
    @unpack MU, lab_func, e, K1, L1 = res

    MU1 = 0
    for j = 1:N
    	for z_index = 1:n_z
    		for a_index = 1:n_a
                K1 = K1 + a_grid[a_index]*MU[j, a_index, z_index]
    			MU1 = MU1 + MU[j, a_index, z_index]
            end
        end
    end
    L1 = 0
    for j = 1:J_retire-1
        for z_index = 1:n_z
            for a_index = 1:n_a
                L1 = L1 + lab_func[j, a_index, z_index]*e[j, z_index]*MU[j, a_index, z_index]
    		end
    	end
    end
end

function solve_model(prim::Primitives, res::Results, θ::Float64 = 0.11, z::Array{Float64, 1} = [3.0, 0.5], γ::Float64 = 0.42, λ::Float64 = 0.5)
    @unpack K0, L0, K1, L1, lab_func = res
    prim, res = Initialize(prim, θ, z, γ)

    update_price(prim, res)

    tol = 1/1000
    i = 0

    while true
        i = i+1
        println("Iteration #", i)
        println("Capital demand: ", K0)
        println("Labor demand: ", L0)
        solve_HH_problem(prim, res)
        asset_dist(prim, res)
        aggregate_lk(prim, res)
        println("Capital supply: ", K1)
        println("Labor supply: ", L1)
        diff = abs(K0 - K1) + abs(L0 - L1)
        println("Absolute difference is: ", diff)
        println("-------------------------------------")

        if diff > tol
            K0 = λ*K1 + (1-λ)*K0
            L0 = λ*L1 + (1-λ)*L0
            update_price(prim, res)
        else
            break
        end
    end

    res
end
