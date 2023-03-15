##

@with_kw struct Primitives
    α::Float64 = -1.0
    β::Float64 = 0.8
    θ::Float64 = 0.64
    A::Float64 = 1/200
    c_f::Float64 = 10.0
    c_e::Float64 = 5.0
    s_grid::Array{Float64, 1} = [3.98e-4; 3.58; 6.82; 12.18; 18.79]
    ns::Int64 = length(s_grid)
    F::Array{Float64} = [0.6598 0.2600 0.0416 0.0331 0.0055;
                         0.1997 0.7201 0.0420 0.0326 0.0056;
                         0.2 0.2 0.5555 0.0344 0.0101;
                         0.2 0.2 0.2502 0.3397 0.0101;
                         0.2 0.2 0.25 0.34 0.01]
    v::Array{Float64} = [0.37; 0.4631; 0.1102; 0.0504; 0.0063]
end

## structure that holds model results

mutable struct Results
    p::Float64
    N_d::Array{Float64}
    profit_func::Array{Float64, 1}
    exit_func::Array{Float64, 1}
    val_func::Array{Float64, 1}
    M::Float64
    μ::Array{Float64}
end

##

function Initialize()
    prim = Primitives()
    ns = prim.ns
    p = 1.0
    N_d = zeros(ns)
    profit_func = zeros(ns)
    exit_func = ones(ns)
    val_func = zeros(ns)
    M = 5.0
    μ = ones(ns)

    res = Results(p, N_d, profit_func, exit_func, val_func, M, μ)
    prim, res
end

##

function labor_profit(prim::Primitives, res::Results, s::Float64)
    @unpack c_f, θ = prim
    @unpack p = res

    n = (θ * p * s) ^ (1 / (1 - θ))
    if n < 0
        n = 0
    end

    profit = p * s * n^θ - n - p * c_f

    n, profit
end

##

function Bellman(prim::Primitives, res::Results)
    @unpack ns, β, F, s_grid = prim
    @unpack val_func, profit_func = res

    exit_func_next = zeros(Int64, ns)
    val_func_next = zeros(ns)

    for s_index = 1:ns
        val_tomorrow = 0.0
        for sp_index = 1:ns
            val_tomorrow += F[s_index, sp_index] * val_func[sp_index]
        end

        if val_tomorrow >= 0
            exit_func_next[s_index] = 0
            val_func_next[s_index] = profit_func[s_index] + β * val_tomorrow
        else
            exit_func_next[s_index] = 1
            val_func_next[s_index] = profit_func[s_index]
        end
    end

    exit_func_next, val_func_next
end

##

function solve_firm_problem(prim::Primitives, res::Results, α::Float64)
    @unpack ns, s_grid = prim

    n = 1
    tol = 0.0001
    err = 100.0

    N_d_next = zeros(ns)
    profit_func_next = zeros(ns)
    for s_index = 1:ns
        N_d_next[s_index], profit_func_next[s_index] = labor_profit(prim, res, s_grid[s_index])
    end
    res.N_d = N_d_next
    res.profit_func = profit_func_next

    while true
        if α == 0.0
            exit_func_next, val_func_next = Bellman(prim, res)
        else
            exit_func_next, val_func_next = Bellman_random(prim, res, α)
        end
        err = maximum(abs.(res.val_func .- val_func_next))
        # println("\n***** ", n, "th iteration *****")
        # @printf("Absolute difference: %0.4f.\n", float(err))
        # println("***************************")
        if err < tol
            break
        end
        res.exit_func = exit_func_next
        res.val_func = val_func_next
        n += 1

    end

    # println("\nFirm problem is solved after ", n, " iterations.\n")
end

##

function stationary_price(prim::Primitives, res::Results, α::Float64)
    @unpack v, c_e = prim

    n = 1                                                                        # count
    err = 100.0
    tol = 0.001
    EC = 100.0
    p_update = 0.0

    while true
        solve_firm_problem(prim, res, α)
        EC = sum(res.val_func .* v)/res.p - c_e
        err = abs(EC)
        if err < tol
            break
        end
        if EC >= 0
            p_update = res.p * 0.9999
        elseif EC < 0
            p_update = res.p * 1.0001
        end
        if n % 100 == 0
            println("************************************")
            @printf("Absolute value of EC (%0.5f) exceeds the tolerance level (%0.5f).\n", float(err), float(tol))
            @printf("=> p is updated from %0.5f to %0.5f.\n\n", float(res.p), float(p_update))
        end
        res.p = p_update
        n += 1
    end

    # println("Stationary p* is found after ", n, " iterations.")
    # @printf("p: %0.4f.", float(res.p))
end

##

function Tstar_Bellman(prim::Primitives, res::Results)
    @unpack ns, F, v = prim
    @unpack exit_func, val_func, μ, M = res

    μ_next = zeros(ns)
    for s_index = 1:ns
        for sp_index = 1:ns
            μ_next[sp_index] += (1 - exit_func[s_index]) * μ[s_index] * F[s_index, sp_index] + (1 - exit_func[s_index]) * F[s_index, sp_index] * M * v[s_index]
        end
    end

    μ_next
end

##

function μ_dist(prim::Primitives, res::Results)

    n = 1                                                                        # count
    tol = 0.0001
    err = 100.0

    while true
        μ_next = Tstar_Bellman(prim, res)
        err = maximum(abs.(μ_next.-res.μ))
        # println("\n***** ", n, "th iteration *****")
        # @printf("Absolute difference: %0.4f.\n", float(err))
        # println("***************************")
        if err < tol
            break
        end
        res.μ = μ_next
        n += 1
    end
    # println("Invariant μ converged in ", n, " iterations")
end

##

function compute_LMC(prim::Primitives, res::Results)
    @unpack A, v = prim
    @unpack N_d, μ, M, profit_func = res

    L_d = sum(N_d .* μ  + N_d * M .* v) # sum(N_d .* μ) #
    Π = sum(μ .* profit_func)
    L_s = 1/A - Π
    LMC = L_d - L_s

    LMC
end

##

function stationary_M(prim::Primitives, res::Results)

    n = 1                                                                        # count
    tol = 0.001
    tol = 0.001
    LMC = 100.0
    M_update = 0.0

    while true
        μ_dist(prim, res)
        LMC = compute_LMC(prim, res)
        err = abs(LMC)
        if err < tol
            break
        end
        if LMC > 0
            M_update = res.M * 0.9999
        elseif LMC <= 0
            M_update = res.M * 1.0001
        end
        if n % 100 == 0
            println("************************************")
            @printf("Absolute value of LMC (%0.5f) exceeds the tolerance level (%0.5f).\n", float(err), float(tol))
            @printf("=> M is updated from %0.5f to %0.5f.\n\n", float(res.M), float(M_update))
        end
        res.M = M_update
        n += 1
    end

    # println("Stationary M* is found after ", n, " iterations.\n")
    # @printf("M: %0.4f.", float(res.M))

end

## solve the model

function solve_model(prim::Primitives, res::Results, α::Float64)
    stationary_price(prim, res, α)
    stationary_M(prim, res)

    res
end

##

function result_table(prim::Primitives, res::Results)
    @unpack v = prim
    @unpack p, exit_func, μ, M, N_d = res

    price = p
    mass_incum = sum(μ .* (1 .- exit_func))
    mass_exit = sum(μ .* exit_func)
    mass_ent = M
    agg_lab = sum(N_d .* μ  + N_d * M .* v)
    lab_inc = sum(N_d .* μ)
    lab_ent = sum(N_d * M .* v)
    frac_lab_ent = lab_ent/agg_lab

    [price, mass_incum, mass_ent, mass_exit, agg_lab, lab_inc, lab_ent, frac_lab_ent]
end

##

function Bellman_random(prim::Primitives, res::Results, α::Float64)
    @unpack ns, β, F = prim
    @unpack profit_func, val_func = res

    exit_func_next = zeros(ns)
    val_func_temp = zeros(ns, 2)
    val_func_next = zeros(ns)

    for s_index = 1:ns
        val_tomorrow = 0.0
        for sp_index = 1:ns
            val_tomorrow += F[s_index, sp_index] * val_func[sp_index]
        end
        val_func_temp[s_index, 1] = profit_func[s_index] + β * val_tomorrow
        val_func_temp[s_index, 2] = profit_func[s_index]
        max_val = maximum([val_func_temp[s_index, 1] val_func_temp[s_index, 2]])
        val_func_next[s_index] = 0.57721 / α + 1 / α * log(exp(α * (val_func_temp[s_index, 1] - max_val)) + exp(α * (val_func_temp[s_index, 2] - max_val))) + max_val
        exit_func_next[s_index] = exp(α * (val_func_temp[s_index, 2] - max_val))/(exp(α * (val_func_temp[s_index, 1] - max_val)) + exp(α * (val_func_temp[s_index, 2] - max_val)))
    end

    exit_func_next, val_func_next
end
