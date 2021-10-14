# ------------------------------------------------------------------------------
# Author: Heejin
# Krusell and Smith (1998, JPE)
# Octobor 14, 2021
# ps5_hj.jl
# ------------------------------------------------------------------------------

using Random, Interpolations, Optim, Distributions

## Initialize

@with_kw struct Primitives
    β::Float64 = 0.99
    α::Float64 = 0.36
    δ::Float64 = 0.025
    λ::Float64 = 0.5

    N::Int64 = 5000 # number of individuals
    T::Int64 = 11000 # number of time periods
    # burn::Int64 = 1000
    #
    # tol_vfi::Float64 = 1e-4
    # tol_coef::Float64 = 1e-4
    # tol_r2::Float64 = 1.0 - 1e-2
    # maxit::Int64 = 10000

    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0 # Duration (Good Times)
    u_b::Float64 = 0.1 # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0 # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b
    pgb::Float64 = 1.0 - (d_g-1.0)/d_g
    pbg::Float64 = 1.0 - (d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pgb00::Float64 = 0.75*pgg00
    pbg00::Float64 = 1.25*pbb00

    #transition probabilities for aggregate states and becoming employed
    pgg10::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb10::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pgb10::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pbg10::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg01::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb01::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pgb01::Float64 = 1.0 - 1.25*pbb00
    pbg01::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pgb11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pbg11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg10
                             pgg01 pgg00]

    Mgb::Array{Float64,2} = [pgb11 pgb10
                             pgb01 pgb00]

    Mbg::Array{Float64,2} = [pbg11 pbg10
                             pbg01 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb10
                              pbb01 pbb00]

    markov::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                                pbg*Mbg pbb*Mbb]

    # grids

    k_lb::Float64 = 0.001
    k_ub::Float64 = 20.0
    n_k::Int64 = 21
    k_grid::Array{Float64,1} = range(k_lb, stop = k_ub, length = n_k)

    K_lb::Float64 = 10.0
    K_ub::Float64 = 15.0
    n_K::Int64 = 11
    K_grid::Array{Float64,1} = range(K_lb, stop = K_ub, length = n_K)

    ē::Float64 = 0.3271
    ε_grid::Array{Float64,1} = [1.0, 0.0] .* ē
    n_ε::Int64 = length(ε_grid)

    z_grid::Array{Float64,1} = [1.01, 0.99]
    n_z::Int64 = length(z_grid)

    L_grid::Array{Float64,1} = [1.0-u_g, 1.0-u_b] .* ē
end

mutable struct Results
    pol_func::Array{Float64,4}
    val_func::Array{Float64,4}
    a0::Float64
    a1::Float64
    b0::Float64
    b1::Float64
    idio_state::Array{Float64}
    agg_state::Array{Float64}
    R²::Float64
end

function Initialize()
    prim = Primitives()
    @unpack n_k, n_K, n_z, n_ε, N, T = prim
    pol_func = zeros(n_k, n_ε, n_K, n_z)
    val_func = zeros(n_k, n_ε, n_K, n_z)
    a0 = 0.0095
    a1 = 0.999
    b0 = 0.0085
    b1 = 0.999
    idio_state = zeros(N, T)
    agg_state = zeros(T)
    R² = 0.0
    res = Results(pol_func, val_func, a0, a1, b0, b1, idio_state, agg_state, R²)

    prim, res
end

function draw_shocks(prim::Primitives)
    @unpack pgg, pbb, Mgg, Mgb, Mbg, Mbb, N, T = prim

    # Shock
    Random.seed!(12032020)
    dist = Uniform(0, 1)

    # Allocate space for shocks and initialize
    idio_state = zeros(N, T)
    agg_state = zeros(T)
    idio_state[:,1] .= 1
    agg_state[1] = 1

    for t = 2:T
        agg_shock = rand(dist)
        if agg_state[t-1] == 1 && agg_shock < pgg
            agg_state[t] = 1
        elseif agg_state[t-1] == 1 && agg_shock > pgg
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock < pbb
            agg_state[t] = 2
        elseif agg_state[t-1] == 2 && agg_shock > pbb
            agg_state[t] = 1
        end

        for n = 1:N
            idio_shock = rand(dist)
            if agg_state[t-1] == 1 && agg_state[t] == 1
                p11 = Mgg[1,1]
                p00 = Mgg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 1 && agg_state[t] == 2
                p11 = Mgb[1,1]
                p00 = Mgb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 1
                p11 = Mbg[1,1]
                p00 = Mbg[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            elseif agg_state[t-1] == 2 && agg_state[t] == 2
                p11 = Mbb[1,1]
                p00 = Mbb[2,2]

                if idio_state[n,t-1] == 1 && idio_shock < p11
                    idio_state[n,t] = 1
                elseif idio_state[n,t-1] == 1 && idio_shock > p11
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock < p00
                    idio_state[n,t] = 2
                elseif idio_state[n,t-1] == 2 && idio_shock > p00
                    idio_state[n,t] = 1
                end
            end
        end
    end

    idio_state, agg_state
end

function Prices(prim::Primitives, K_index::Int64, z_index::Int64)
    @unpack α, z_grid, K_grid, L_grid = prim

    r = α*z_grid[z_index]*(K_grid[K_index]/L_grid[z_index])^(α-1)
    w = (1-α)*z_grid[z_index]*(K_grid[K_index]/L_grid[z_index])^α

    r, w
end


##scale

function Bellman(prim::Primitives, res::Results)
    @unpack β, α, δ, n_k, k_grid, n_ε, ε_grid, K_grid, n_K, n_z, z_grid, u_g, u_b, markov = prim
    @unpack val_func, a0, a1, b0, b1= res

    val_func_update = zeros(n_k, n_ε, n_K, n_z)

    k_interp = interpolate(k_grid, BSpline(Linear()))
    v_interp = interpolate(val_func, BSpline(Linear()))

    for (K_index, K_today) in enumerate(K_grid), (z_index, z_today) in enumerate(z_grid)
        if z_index == 1
            K_tomorrow = a0 + a1*log(K_today)
        elseif z_index == 2
            K_tomorrow = b0 + b1*log(K_today)
        end
        K_tomorrow = exp(K_tomorrow)

        # See that K_tomorrow likely does not fall on our K_grid...this is why we need to interpolate!
        Kp_index = get_index(K_tomorrow, K_grid)

        r_today, w_today = Prices(prim, K_index, z_index)

        for (ε_index, ε_today) in enumerate(ε_grid)
            row = ε_index + 2*(z_index-1)

            for (k_index, k_today) in enumerate(k_grid)
                budget_today = r_today*k_today + w_today*ε_today + (1.0 - δ)*k_today

                # We are defining the continuation value. Notice that we are interpolating over k and K.
                v_tomorrow(kp_index) = markov[row, 1]*v_interp(kp_index, 1, Kp_index, 1) + markov[row, 2]*v_interp(kp_index, 2, Kp_index, 1) +
                                    markov[row, 3]*v_interp(kp_index, 1, Kp_index, 2) + markov[row, 4]*v_interp(kp_index, 2, Kp_index, 2)


                # We are now going to solve the HH's problem (solve for k).
                # We are defining a function val_func as a function of the agent's capital choice.
                val(kp_index) = log(budget_today - k_interp(kp_index)) +  β*v_tomorrow(kp_index)

                # Need to make our "maximization" problem a "minimization" problem.
                obj(kp_index) = -val(kp_index)
                lower = 1.0
                upper = get_index(budget_today, k_grid)

                # Then, we are going to maximize the value function using an optimization routine.
                # Note: Need to call in optimize to use this package.
                opt = optimize(obj, lower, upper)

                k_tomorrow = k_interp(opt.minimizer[1])
                v_today = -opt.minimum

                # Update PFs
                res.pol_func[k_index, ε_index, K_index, z_index] = k_tomorrow
                val_func_update[k_index, ε_index, K_index, z_index] = v_today
            end
        end
    end

    val_func_update
end


function V_iterate(prim::Primitives, res::Results)
    error = 100.0
    tol = 0.0001
    i = 0
    while true
        i += 1
        val_func_update = Bellman(prim, res) #get new guess of value function
        error = maximum(abs.(res.val_func - val_func_update))/abs(maximum(val_func_update)) #update error
        res.val_func = val_func_update #update value function
        if error < tol
            break
        end
    end
    println("  ")
    println("********************************")
    println("Value function iteration is done.")
    println("Number of iterations: ", i, ".")
end

function get_index(val::Float64, grid::Array{Float64,1})
    n = length(grid)
    index = 0
    if val <= grid[1]
        index = 1
    elseif val >= grid[n]
        index = n
    else
        index_upper = findfirst(x->x>val, grid)
        index_lower = index_upper - 1
        val_upper, val_lower = grid[index_upper], grid[index_lower]

        index = index_lower + (val - val_lower) / (val_upper - val_lower)
    end
    return index
end


function Solve_model(prim::Primitives, res::Results; tol::Float64=.01)
    error = 100.0
    tol = 0.01
    i = 0
    while true
        i += 1

        idio_state, agg_state = draw_shocks(prim)
        println("Shock drawing is done.")

        V_iterate(prim, res)
        println("Value function iteration is done.")

        error = Estimate_regression(prim, res)

        val_func_update = Bellman(prim, res) #get new guess of value function
        error = maximum(abs.(res.val_func - val_func_update))/abs(maximum(val_func_update)) #update error
        res.val_func = val_func_update #update value function
        if error < tol
            break
        end
    end
    # println("  ")
    # println("********************************")
    # println("Value function iteration is done.")
    # println("Number of iterations: ", i, ".")
end
