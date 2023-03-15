# ------------------------------------------------------------------------------
# Author: Heejin
# Conesa and Krueger (1999, RED)
# Octobor 11, 2021
# ps4_model_transition_hj.jl
# ------------------------------------------------------------------------------

using Parameters

## Initialize

mutable struct Path_transition
    θ::Array{Float64}                                                            # path of income tax rate
    K0_path::Array{Float64, 1}                                                   # transition path of aggregate capital
    L0_path::Array{Float64, 1}                                                   # transition path of aggregate labor
    w_path::Array{Float64, 1}                                                    # transition path of wage
    r_path::Array{Float64, 1}                                                    # transition path of interest rate
    b_path::Array{Float64, 1}                                                    # transition path of social security benefit
    val_func_path::Array{Float64}                                                # transition path of value function
    pol_func_path::Array{Float64}                                                # transition path of policy function
    lab_func_path::Array{Float64}                                                # transition path of labor function
    μ_path::Array{Float64}                                                       # transition path of asset distribution
end

function Initialize_transition(T::Int64, t_policy::Int64)                        # initialize (argument: total transition period & policy implmentation time)
    @unpack α, δ, J_retire, N, n_a, n_z = prim

    θ = [repeat([0.11], t_policy); repeat([0.0], T-t_policy)]
    K0_path = collect(range(bm_ss.K0, stop = bm_no_ss.K0, length = T))
    L0_path = collect(range(bm_ss.L0, stop = bm_no_ss.L0, length = T))
    μ_path = zeros(N, n_a, n_z, T)
    μ_path[:, :, :, 1] = bm_ss.μ
    w_path = (1 - α) .* (K0_path .^ α) .* (L0_path .^ (- α))
    r_path = α .* (K0_path .^ (α - 1)) .* (L0_path .^ (1-α)) .- δ
    b_path = (θ .* w_path .* L0_path) ./ sum(μ_path[J_retire:N, :, :, 1])
    val_func_path = zeros(N, n_a, n_z, T)
    val_func_path[:, :, :, T] = bm_no_ss.val_func
    pol_func_path = zeros(N, n_a, n_z, T)
    pol_func_path[:, :, :, T] = bm_no_ss.pol_func
    lab_func_path = zeros(N, n_a, n_z, T)
    lab_func_path[:, :, :, T] = bm_no_ss.lab_func
    path = Path_transition(θ, K0_path, L0_path, w_path, r_path ,b_path, val_func_path, pol_func_path, lab_func_path, μ_path)
end

## HH Optimization Problem

function Bellman_retired_transition(prim::Primitives, res::Results, path::Path_transition, T::Int64)
    @unpack β, N, J_retire, n_a, σ, γ, a_grid, n_z = prim
    for t_index = (T-1):-1:1
        for a_index=1:n_a
            c = (1+path.r_path[t_index])*a_grid[a_index]+path.b_path[t_index]
            path.val_func_path[N, a_index, 1, t_index] = u_retired(prim, res, c)
        end
        for j = (N-1):-1:J_retire
            ap_index_min = 1
            for a_index = 1:n_a
                budget = (1 + path.r_path[t_index]) * a_grid[a_index] + path.b_path[t_index]
                candidate_max = -Inf
                for ap_index = ap_index_min:n_a
                    val = u_retired(prim, res, budget - a_grid[ap_index]) + β * path.val_func_path[j+1,ap_index,1,t_index+1]
                    if val < candidate_max
                        path.val_func_path[j, a_index, 1, t_index] = candidate_max
                        path.pol_func_path[j, a_index, 1, t_index] = a_grid[ap_index-1]
                        ap_index_min = ap_index-1
                        break
                    elseif ap_index==n_a
                        path.val_func_path[j, a_index, 1, t_index] = val
                        path.pol_func_path[j, a_index, 1, t_index] = a_grid[ap_index]
                    end
                    candidate_max = val
                end
            end
        end
    end
    path.pol_func_path[:, :, 2, 1:T-1] = path.pol_func_path[:, :, 1, 1:T-1]
    path.val_func_path[:, :, 2, 1:T-1] = path.val_func_path[:, :, 1, 1:T-1]
end

function optimal_labor_transition(prim::Primitives, res::Results, path::Path_transition, a::Float64, ap::Float64, j::Int64, z_index::Int64, t_index::Int64)
    @unpack r_path, w_path, θ = path
    @unpack γ, e = prim
    l::Float64 = (γ*(1-θ[t_index])*e[j,z_index]*w_path[t_index] - (1-γ)*((1+r_path[t_index])*a - ap)) / ((1-θ[t_index])*w_path[t_index]*e[j,z_index])
    if l>1
        l = 1
    elseif l<0
        l = 0
    end
    l
end

function Bellman_worker_transition(prim::Primitives, res::Results, path::Path_transition, T::Int64)
    @unpack β, Π, J_retire, n_a, σ, γ, e, n_a, a_grid, n_z = prim
    @unpack θ = path
    for t_index = (T-1):-1:1
        for j = (J_retire-1):-1:1
            for z_index = 1:n_z
                ap_index_min = 1
                for a_index = 1:n_a
                    candidate_max =  -Inf
                    for ap_index = ap_index_min:n_a
                        l = optimal_labor_transition(prim, res, path, a_grid[a_index], a_grid[ap_index], j, z_index, t_index)
                        c = path.w_path[t_index] * (1-θ[t_index]) * e[j,z_index]*l + (1+path.r_path[t_index])*a_grid[a_index]-a_grid[ap_index]
                        val = u_worker(prim, res, c, l)
                        for zp_index = 1:n_z
                            val += β*Π[z_index,zp_index]*path.val_func_path[j+1,ap_index,zp_index, t_index+1]
                        end
                        if val < candidate_max
                            path.val_func_path[j, a_index, z_index, t_index] = candidate_max
                            path.pol_func_path[j, a_index, z_index, t_index] = a_grid[ap_index-1]
                            path.lab_func_path[j, a_index, z_index, t_index] = optimal_labor_transition(prim, res, path, a_grid[a_index], a_grid[ap_index-1], j, z_index, t_index)
                            ap_index_min = ap_index-1
                            break
                        elseif ap_index==n_a
                            path.val_func_path[j, a_index, z_index, t_index] = val
                            path.pol_func_path[j, a_index, z_index, t_index] = a_grid[ap_index]
                            path.lab_func_path[j, a_index, z_index, t_index] = l
                        end
                        candidate_max = val
                    end
                end
            end
        end
    end
end

function solve_HH_problem_transition(prim::Primitives, res::Results, path::Path_transition, T::Int64)
    Bellman_retired_transition(prim, res, path, T)
    Bellman_worker_transition(prim, res, path, T)
end

## Steady-State Distribution Path

function μ_distribution_path(prim::Primitives, res:: Results, path::Path_transition, T::Int64)
    @unpack μ_path, pol_func_path = path
    @unpack a_grid, N, n, n_a, n_z, Π₀, Π = prim
    μ_path = zeros(N, n_a, n_z, T)
    μ_path[:, :, :, 1] = bm_ss.μ
    for t_index = 2:T
        μ_path[1, 1, :, t_index] = μ_path[1, 1, :, 1]
    end
    for t_index = 1:T-1
        for j = 1:(N-1)
            for a_index=1:n_a, z_index=1:n_z
                if μ_path[j, a_index, z_index, t_index] == 0
                    continue
                end
                ap_choice = pol_func_path[j, a_index, z_index, t_index]
                for ap_index=1:n_a
                    if a_grid[ap_index] == ap_choice
                        for zp_index=1:n_z
                            μ_path[j+1, ap_index, zp_index, t_index+1] += Π[z_index, zp_index]*μ_path[j, a_index, z_index, t_index]./(1+n)
                        end
                    end
                end
            end
        end
    end
    path.μ_path = μ_path
end

function aggregate_L_path(prim::Primitives, res:: Results, path::Path_transition, T::Int64)
    @unpack μ_path, lab_func_path = path
    @unpack n_a, n_z, N, e, a_grid = prim
    L1 = zeros(T)
    L1[1] = bm_ss.L0
    E = zeros(N, n_a, n_z)
    for a_index=1:n_a, z_index=1:n_z
        E[1:45, a_index, z_index] = e[:,z_index]
    end
    for t_index=1:T
        L1[t_index] = sum(μ_path[:, :, :, t_index].*lab_func_path[:, :, :, t_index].*E)
    end
    L1
end

function aggregate_K_path(prim::Primitives, res:: Results, path::Path_transition, T::Int64)
    @unpack μ_path = path
    @unpack n_a, n_z, N, e, a_grid = prim
    K1 = zeros(T)
    K1[1] = bm_ss.K0
    A_grid = zeros(N, n_a, n_z)
    for j=1:N, z_index=1:n_z
        A_grid[j, :, z_index] = a_grid
    end
    for t_index=1:T
        K1[t_index] = sum(μ_path[:, :, :, t_index].*A_grid)
    end
    K1
end

function Reinitialize_transition(prim::Primitives, path::Path_transition, K1_path::Array{Float64}, L1_path::Array{Float64}, T::Int64)
    @unpack K0_path, L0_path, θ = path
    @unpack α, δ, J_retire, N, n_a, n_z = prim
    λ::Float64 = 0.5
    path.K0_path = λ .* K1_path .+ (1 - λ) .* K0_path
    path.L0_path = λ .* L1_path .+ (1 - λ) .* L0_path
    path.μ_path = zeros(N, n_a, n_z, T)
    path.μ_path[:, :, :, 1] = bm_ss.μ
    path.w_path = (1 - α) .* (path.K0_path .^ α) .* (path.L0_path .^ (- α))
    path.r_path = α .* (path.K0_path .^ (α - 1)) .* (path.L0_path .^ (1-α)) .- δ
    path.b_path = (θ .* path.w_path .* path.L0_path) ./ sum(path.μ_path[J_retire:N, :, :, 1])
    path.val_func_path = zeros(N, n_a, n_z, T)
    path.val_func_path[:, :, :, T] = bm_no_ss.val_func
    path.pol_func_path = zeros(N, n_a, n_z, T)
    path.pol_func_path[:, :, :, T] = bm_no_ss.pol_func
    path.lab_func_path = zeros(N, n_a, n_z, T)
    path.lab_func_path[:, :, :, T] = bm_no_ss.lab_func
end

## Solve the Model

function solve_model_transition(prim::Primitives, res::Results, t::Int64)
    @unpack α, δ, J_retire, N, n_a, n_z = prim
    T_delta = 20
    T = 30
    t_policy = t
    tol = 0.005
    i = 0
    λ = 0.5
    K1_path = zeros(T)
    L1_path = zeros(T)
    path = Initialize_transition(T, t_policy)
    while true
        path = Initialize_transition(T, t_policy)
        while true
            i += 1
            println("***********************************")
            println("Iteration #", i)
            solve_HH_problem_transition(prim, res, path, T)
            println("solve_HH_problem_transition is done.")
            μ_distribution_path(prim, res, path, T)
            println("μ_distribution_path is done.")
            K1_path = aggregate_K_path(prim, res, path, T)
            L1_path = aggregate_L_path(prim, res, path, T)
            println("aggregate_LK_path is done.")
            display(plot([path.K0_path K1_path repeat([bm_ss.K0], T) repeat([bm_no_ss.K0], T)],
                         label = ["K Demand" "K Supply" "Stationary K w/ SS" "Stationary K w/o SS"],
                         title = "Capital Transition Path", legend = :bottomright))
            diff = maximum(abs.(path.K0_path .- K1_path)./K1_path) + maximum(abs.(path.L0_path .- L1_path)./L1_path)
            println("difference: ", diff)
            if diff > tol
                Reinitialize_transition(prim, path, K1_path, L1_path, T)
            else
                println("***********************************")
                println("★ K path has been converged. ★  ")
                println(" ")
                break
            end
        end
        error = abs(path.K0_path[T] - bm_no_ss.K0)/bm_no_ss.K0
        if error > tol
            println("************************************")
            println("Error (", error, ") exceeds the tolerance level (", tol, ").")
            println("=> Length of transition path has increased from ", T, " to ", T+T_delta, ".")
            T += T_delta
        else
            println("***********************************")
            println("Error (", error, ") is below the tolerance level (", tol, ").")
            println("★ Iteration is done after ", i, " iterations. ★  ")
            println(" ")
            break
        end
    end
    path, T
end

## Equivalent Variation and Vote Share

function Equivalent_variation(prim::Primitives, res::Results, path::Path_transition)
    @unpack N, n_a, n_z, γ, σ = prim
    EV = zeros(N, n_a, n_z)
    EV = (path.val_func_path[:, :, :, 2] ./ bm_ss.val_func).^(1/(γ * (1 - σ)))
    EVⱼ = zeros(N)
    for j=1:N
        EVⱼ[j] = sum(EV[j, :, :] .* bm_ss.μ[j, :, :])
    end
    EVⱼ, EV
end

function Vote_share(EV::Array{Float64})
    Share::Float64 = sum((EV.>=1).*bm_ss.μ)
end
