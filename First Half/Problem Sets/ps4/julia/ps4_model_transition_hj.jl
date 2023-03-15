## Initialize

mutable struct Path
    θ::Array{Float64}                                                            # path of income tax rate
    K_path::Array{Float64, 1}                                                    # transition path of aggregate capital
    L_path::Array{Float64, 1}                                                    # transition path of aggregate labor
    w_path::Array{Float64, 1}                                                    # transition path of wage
    r_path::Array{Float64, 1}                                                    # transition path of interest rate
    b_path::Array{Float64, 1}                                                    # transition path of social security benefit
    val_func_path::Array{Float64, 4}                                             # transition path of value function
    pol_func_path::Array{Float64, 4}                                             # transition path of policy function
    lab_func_path::Array{Float64, 4}                                             # transition path of labor function
    F_path::Array{Float64, 4}                                                    # transition path of asset distribution
end

##

function Initialize_trans(T::Int64, t_policy::Int64)                        # initialize (argument: total transition period & policy implmentation time)
    @unpack α, δ, J_retire, N, na, nz = prim

    θ = [repeat([0.11], t_policy); repeat([0.0], T-t_policy)]
    F_path = zeros(N, na, nz, T)
    F_path[:, :, :, 1] = bm_ss.F
    K_path = collect(range(bm_ss.K, stop = bm_wo_ss.K, length = T))
    L_path = collect(range(bm_ss.L, stop = bm_wo_ss.L, length = T))
    w_path = (1 - α) .* (K_path .^ α) .* (L_path .^ (- α))
    r_path = α .* (K_path .^ (α - 1)) .* (L_path .^ (1-α)) .- δ
    b_path = (θ .* w_path .* L_path) ./ sum(F_path[J_retire:N, :, :, 1])
    val_func_path = zeros(N, na, nz, T)
    pol_func_path = zeros(N, na, nz, T)
    lab_func_path = zeros(N, na, nz, T)
    path = Path(θ, K_path, L_path, w_path, r_path ,b_path, val_func_path, pol_func_path, lab_func_path, F_path)
end

##

function optimal_labor_trans(prim::Primitives, res::Results, path::Path, a::Float64, ap::Float64, j::Int64, z_index::Int64, t_index::Int64)
    @unpack r_path, w_path, θ = path
    @unpack γ, e, J_retire = prim

    l = (γ * (1 - θ[t_index]) * e[j,z_index] * w_path[t_index] - (1 - γ) * ((1 + r_path[t_index]) * a - ap)) / ((1 - θ[t_index]) * w_path[t_index] * e[j, z_index])
    if l>1
        l = 1.0
    elseif l<0
        l = 0.0
    end
    l
end

## HH Optimization Problem

function Bellman_path(prim::Primitives, res::Results, path::Path, T::Int64)
    @unpack β, N, J_retire, na, σ, γ, a_grid, nz, e, Π = prim
    @unpack r_path, b_path, w_path, θ = path

    val_func_path = zeros(N, na, nz, T)
    pol_func_path = zeros(N, na, nz, T)
    lab_func_path = zeros(N, na, nz, T)
    val_func_path[:, :, :, T] = bm_wo_ss.val_func
    pol_func_path[:, :, :, T] = bm_wo_ss.pol_func
    lab_func_path[:, :, :, T] = bm_wo_ss.lab_func

    for t_index = (T-1):-1:1
        for a_index=1:na
            c = (1 + r_path[t_index]) * a_grid[a_index] + b_path[t_index]
            val_func_path[N, a_index, 1, t_index] = utility(prim, res, c, 0.0, N)
        end
        for j = (N-1):-1:J_retire
            for a_index = 1:na
                cand_max = -Inf
                ap_index_min = 1
                for ap_index = ap_index_min:na
                    c = (1 + r_path[t_index]) * a_grid[a_index] + b_path[t_index] - a_grid[ap_index]
                    val = utility(prim, res, c, 0.0, j) + β * val_func_path[j+1, ap_index, 1, t_index+1]
                    if val < cand_max
                        break
                    end
                    cand_max = val
                    pol_func_path[j, a_index, 1, t_index] = a_grid[ap_index]
                    ap_index_min = ap_index
                end
                val_func_path[j, a_index, 1, t_index] = cand_max
            end
        end
        val_func_path[:, :, 2, t_index] = val_func_path[:, :, 1, t_index]
        pol_func_path[:, :, 2, t_index] = pol_func_path[:, :, 1, t_index]
        lab_func_path[:, :, 2, t_index] = lab_func_path[:, :, 1, t_index]

        for j = (J_retire - 1):-1:1
            for z_index = 1:nz
                for a_index = 1:na
                    cand_max = -Inf
                    ap_index_min = 1
                    for ap_index = ap_index_min:na
                        l = optimal_labor_trans(prim, res, path, a_grid[a_index], a_grid[ap_index], j, z_index, t_index)
                        c = w_path[t_index] * (1 - θ[t_index]) * e[j, z_index] * l + (1 + r_path[t_index]) * a_grid[a_index] - a_grid[ap_index]
                        val = utility(prim, res, c, l, j)
                        for zp_index = 1:nz
                            val += β * Π[z_index, zp_index] * val_func_path[j+1, ap_index, zp_index, t_index + 1]
                        end
                        if val < cand_max
                            break
                        end
                        lab_func_path[j, a_index, z_index, t_index] = l
                        pol_func_path[j, a_index, z_index, t_index] = a_grid[ap_index]
                        ap_index_min = ap_index
                        cand_max = val
                    end
                    val_func_path[j, a_index, z_index, t_index] = cand_max
                end
            end
        end
        @printf("HH problem: %0.2f percent is done.", float((T - t_index)/(T - 1) * 100))
        println("")
    end
    val_func_path, pol_func_path, lab_func_path
end

## Steady-State Distribution Path

function F_dist_path(prim::Primitives, res::Results, path::Path, T::Int64)
    @unpack pol_func_path = path
    @unpack a_grid, N, n, na, nz, Π₀, Π = prim

    F_path = zeros(N, na, nz, T)
    F_path[:, :, :, 1] = bm_ss.F

    for t_index = 2:T
        F_path[1, 1, :, t_index] = F_path[1, 1, :, 1]
    end

    for t_index = 1:T-1
        for j = 1:(N-1)
            for a_index = 1:na, z_index = 1:nz
                if F_path[j, a_index, z_index, t_index] > 0
                    ap = pol_func_path[j, a_index, z_index, t_index]
                    ap_index = argmin(abs.(ap .- a_grid))
                    for zp_index = 1:nz
                        F_path[j+1, ap_index, zp_index, t_index+1] += Π[z_index, zp_index]*F_path[j, a_index, z_index, t_index]./(1+n)
                    end
                end
            end
        end
    end
    F_path
end

##

function aggregate_L_path(prim::Primitives, res:: Results, path::Path, T::Int64)
    @unpack F_path, lab_func_path = path
    @unpack na, nz, N, e, a_grid = prim

    L_path = zeros(T)
    L_path[1] = bm_ss.L
    E_grid = zeros(N, na, nz)

    for a_index = 1:na, z_index = 1:nz
        E_grid[1:45, a_index, z_index] = e[:, z_index]
    end
    for t_index = 1:T
        L_path[t_index] = sum(F_path[:, :, :, t_index] .* lab_func_path[:, :, :, t_index] .* E_grid)
    end
    L_path
end

##

function aggregate_K_path(prim::Primitives, res:: Results, path::Path, T::Int64)
    @unpack F_path = path
    @unpack na, nz, N, e, a_grid = prim

    K_path = zeros(T)
    K_path[1] = bm_ss.K
    A_grid = zeros(N, na, nz)

    for j = 1:N, z_index = 1:nz
        A_grid[j, :, z_index] = a_grid
    end

    for t_index = 1:T
        K_path[t_index] = sum(F_path[:, :, :, t_index] .* A_grid)
    end
    K_path
end

##

function adjust_path(prim::Primitives, path::Path, K1_path::Array{Float64}, L1_path::Array{Float64}, T::Int64)
    @unpack K_path, L_path, θ = path
    @unpack α, δ, J_retire, N, na, nz = prim

    λ = 0.5
    K_path_update = λ .* K1_path .+ (1 - λ) .* K_path
    L_path_update = λ .* L1_path .+ (1 - λ) .* L_path
    F_path_update = zeros(N, na, nz, T)
    F_path_update[:, :, :, 1] = bm_ss.F
    w_path_update = (1 - α) .* (K_path_update .^ α) .* (L_path_update .^ (- α))
    r_path_update = α .* (K_path_update .^ (α - 1)) .* (L_path_update .^ (1-α)) .- δ
    b_path_update = (θ .* w_path_update .* L_path_update) ./ sum(F_path_update[J_retire:N, :, :, 1])
    path.K_path = K_path_update
    path.L_path = L_path_update
    path.F_path = F_path_update
    path.w_path = w_path_update
    path.r_path = r_path_update
    path.b_path = b_path_update
end

## Solve the Model

function solve_model_trans(prim::Primitives, res::Results, t::Int64, T::Int64)
    @unpack α, δ, J_retire, N, na, nz = prim
    T_delta = 20
    tol = 0.01
    i = 0
    λ = 0.5
    path = Initialize_trans(T, t)
    while true
        while true
            i += 1
            println("***********************************")
            println("Trial #", i)
            path.val_func_path, path.pol_func_path, path.lab_func_path = Bellman_path(prim, res, path, T)
            println("HH problem is solved.")
            path.F_path = F_dist_path(prim, res, path, T)
            println("F distribution is calculated.")
            K1_path = aggregate_K_path(prim, res, path, T)
            L1_path = aggregate_L_path(prim, res, path, T)
            println("Aggregate L&K calculation is done.")

            display(plot([path.K_path K1_path repeat([bm_ss.K], T) repeat([bm_wo_ss.K], T)],
                         label = ["K Guess" "K Path" "Stationary K w/ SS" "Stationary K w/o SS"],
                         title = "Capital Transition Path", legend = :bottomright))
            diff = maximum(abs.(path.K_path .- K1_path)./K1_path) + maximum(abs.(path.L_path .- L1_path)./L1_path)
            @printf("Difference: %0.3f.", float(diff))
            println("")

            if diff > tol
                adjust_path(prim, path, K1_path, L1_path, T)
                println("K and L paths are adjusted.")
            else
                println("***********************************")
                println("★ K path has been converged. ★  ")
                println(" ")
                break
            end
        end
        error = abs(path.K_path[T] - bm_wo_ss.K)/bm_wo_ss.K
        if error > tol
            println("************************************")
            @printf("Error ( %0.3f ) exceeds the tolerance level ( %0.3f ).", float(error), float(tol))
            println(" ")
            @printf("=> Length of transition path has increased from %0.0f to %0.0f.", Int(T), Int(T+T_delta))
            println(" ")
            T += T_delta
            path = Initialize_trans(T, t)
        else
            println("***********************************")
            @printf("Error ( %0.3f ) is below the tolerance level ( %0.3f ).", float(error), float(tol))
            println(" ")
            println("★ Iteration is done after ", i, " iterations. ★  ")
            println(" ")
            break
        end
    end
    path, T

end

## Equivalent Variation and Vote Share

function Equivalent_variation(prim::Primitives, res::Results, path::Path)
    @unpack N, na, nz, γ, σ = prim
    EV = zeros(N, na, nz)
    EV = (path.val_func_path[:, :, :, 1] ./ bm_ss.val_func).^(1/(γ * (1 - σ)))
    EV_counterfact = (bm_wo_ss.val_func ./ bm_ss.val_func).^(1/(γ * (1 - σ)))
    EV_j = zeros(N)
    EV_j_counterfact = zeros(N)
    for j=1:N
        EV_j[j] = sum(EV[j, :, :] .* bm_ss.F[j, :, :]) ./ sum(bm_ss.F[j, :, :])
        EV_j_counterfact[j] = sum(EV_counterfact[j, :, :] .* bm_ss.F[j, :, :]) ./ sum(bm_ss.F[j, :, :])
    end
    EV_j, EV_j_counterfact, EV, EV_counterfact
end

function Vote_share(EV::Array{Float64})
    vote_share = sum((EV.>=1).*bm_ss.F)
end
