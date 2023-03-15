##


## Initialize

@with_kw struct Primitives
    β::Float64 = 0.99
    α::Float64 = 0.36
    δ::Float64 = 0.025
    λ::Float64 = 0.5
    N::Int64 = 5000 # number of individuals
    T::Int64 = 11000 # number of time periods
    burn::Int64 = 1000 # deal with initial condition dependence

    du_g::Float64 = 1.5 # duration of unemployment spells in good time
    du_b::Float64 = 2.5 # duration of unemployment spells in bad time
    u_g::Float64 = 0.04 # unemployment rate in good time
    u_b::Float64 = 0.1 # unemployment rate in bad time
    d_g::Float64 = 8.0 # duration of good time
    d_b::Float64 = 8.0 # duration of bad time

    k̲::Float64 = 0.001
    k̄::Float64 = 20.0
    nk::Int64 = 21
    k_grid::Array{Float64, 1} = collect(range(k̲, stop = k̄, length = nk))
    K̲::Float64 = 10.0
    K̄::Float64 = 15.0
    nK::Int64 = 11
    K_grid::Array{Float64, 1} = collect(range(K̲, stop = K̄, length = nK))
    ē::Float64 = 0.3271
    ε_grid::Array{Float64, 1} = [1.0, 0.0]
    nε::Int64 = length(ε_grid)
    z_grid::Array{Float64,1} = [1.01, 0.99]
    nz::Int64 = length(z_grid)
    L_grid::Array{Float64,1} = [1.0 - u_g, 1.0 - u_b]

    #transition probabilities for aggregate states
    p_gg::Float64 = (d_g - 1.0) / d_g  # d_g = 1/(1-p_gg) ⇒ p_gg = (d_g - 1.0)/d_g
    p_bb::Float64 = (d_b - 1.0) / d_b  # d_b = 1/(1-p_bb) ⇒ p_bb = (d_b - 1.0)/d_b
    p_gb::Float64 = 1.0 - p_gg
    p_bg::Float64 = 1.0 - p_bb

    #transition probabilities for aggregate states and staying unemployed
    p_gg00_p_gg::Float64 = (du_g - 1.0) / du_g # probability of unemployed → unemployed, conditional on good time → good time
    p_bb00_p_bb::Float64 = (du_b - 1.0) / du_b # probability of unemployed → unemployed, conditional on bad time → bad time
    p_gb00_p_gb::Float64 = 1.25 * p_bb00_p_bb # probability of unemployed → unemployed, conditional on good time → bad time
    p_bg00_p_bg::Float64 = 0.75 * p_gg00_p_gg # probability of unemployed → unemployed, conditional on bad time → good time

    p_gg01_p_gg::Float64 = 1.0 - p_gg00_p_gg
    p_bb01_p_bb::Float64 = 1.0 - p_bb00_p_bb
    p_gb01_p_gb::Float64 = 1.0 - p_gb00_p_gb
    p_bg01_p_bg::Float64 = 1.0 - p_bg00_p_bg

    #transition probabilities for aggregate states and becoming unemployed
    p_gg10_p_gg::Float64 = u_g * (1 - p_gg00_p_gg) / (1.0 - u_g) # u_g = u_g * p_gg00_p_gg + (1 - u_g) * p_gg10_p_gg
    p_bb10_p_bb::Float64 = u_b * (1 - p_bb00_p_bb) / (1.0 - u_b) # u_b = u_b * p_bb00_p_bb + (1 - u_b) * p_bb10_p_bb
    p_gb10_p_gb::Float64 = (u_b - u_g * p_gb00_p_gb) / (1.0 - u_g) # u_b = u_g * p_gb00_p_gb + (1 - u_g) * p_gb10_p_gb
    p_bg10_p_bg::Float64 = (u_g - u_b * p_bg00_p_bg) / (1.0 - u_b) # u_g = u_b * p_bg00_p_bg + (1 - u_b) * p_bg10_p_bg

    p_gg11_p_gg::Float64 = 1.0 - p_gg10_p_gg
    p_bb11_p_bb::Float64 = 1.0 - p_bb10_p_bb
    p_gb11_p_gb::Float64 = 1.0 - p_gb10_p_gb
    p_bg11_p_bg::Float64 = 1.0 - p_bg10_p_bg

    # Markov Transition Matrix
    Π_gg::Array{Float64, 2} = [p_gg11_p_gg p_gg10_p_gg; p_gg01_p_gg p_gg00_p_gg]
    Π_gb::Array{Float64, 2} = [p_gb11_p_gb p_gb10_p_gb; p_gb01_p_gb p_gb00_p_gb]
    Π_bg::Array{Float64, 2} = [p_bg11_p_bg p_bg10_p_bg; p_bg01_p_bg p_bg00_p_bg]
    Π_bb::Array{Float64, 2} = [p_bb11_p_bb p_bb10_p_bb; p_bb01_p_bb p_bb00_p_bb]

    Π::Array{Float64,2} = [p_gg * Π_gg p_gb * Π_gb; p_bg * Π_bg p_bb * Π_bb] # [p_gg11 p_gg10 p_gb11 p_gb10; p_gg01 p_gg00 p_gb01 p_gb00; p_bg11 p_bg10 p_bb11 p_bb10; p_bg01 p_bg00 p_bb01 p_bb00]

    p_g = p_bg / (1 - p_gg + p_bg)  # p_g = p_b * p_bg + p_g * p_gg = (1 - p_g) * p_bg + p_g * p_gg
    L_ss = L_grid[1] * ē * p_g + L_grid[2] * ē * (1 - p_g)
    K_ss = (α / (1/β + δ - 1))^(1/(1 - α)) * L_ss
end

##

mutable struct Results
    pol_func::Array{Float64, 4}
    val_func::Array{Float64, 4}
    a_0::Float64
    b_0::Float64
    a_1::Float64
    b_1::Float64
    state_space_idio::Array{Int64, 2}
    state_space_agg::Array{Int64, 1}
    K_path::Array{Float64, 1}
    k_path::Array{Float64, 2}
    R_squared::Array{Float64, 1}
end

##

function Initialize()
    prim = Primitives()
    @unpack nk, nK, nz, nε, N, T = prim

    pol_func = zeros(nk, nε, nK, nz)
    val_func = zeros(nk, nε, nK, nz)
    a_0 = 0.095 # 0.0
    a_1 = 0.999 # 1.0
    b_0 = 0.085 # 0.0
    b_1 = 0.999 # 1.0
    state_space_idio = zeros(Int64, N, T)
    state_space_agg = zeros(Int64, T)
    K_path = zeros(T)
    k_path = zeros(N, T)
    R_squared = zeros(2)
    res = Results(pol_func, val_func, a_0, b_0, a_1, b_1, state_space_idio, state_space_agg, K_path, k_path, R_squared)

    prim, res
end

##

function draw_shocks(prim::Primitives, res::Results)
    @unpack Π_gg, Π_gb, Π_bg, Π_bb, p_gg, p_bb, N, T = prim

    Random.seed!(1234)
    dist = Uniform(0, 1)

    state_space_idio = zeros(N, T)
    state_space_idio[:, 1] .= 1
    state_space_agg = zeros(T)
    state_space_agg[1] = 1 # 1: good, 2: bad

    for t = 2:T
        shock_agg = rand(dist)
        if state_space_agg[t - 1] == 1 && shock_agg <= p_gg
            state_space_agg[t] = 1
        elseif state_space_agg[t - 1] == 1 && shock_agg > p_gg
            state_space_agg[t] = 2
        elseif state_space_agg[t - 1] == 2 && shock_agg <= p_bb
            state_space_agg[t] = 2
        elseif state_space_agg[t - 1] == 2 && shock_agg > p_bb
            state_space_agg[t] = 1
        end
    end

    for t = 2:T
        if state_space_agg[t - 1] == 1 && state_space_agg[t] == 1
            p_11 = Π_gg[1, 1]
            p_00 = Π_gg[2, 2]
        elseif state_space_agg[t - 1] == 1 && state_space_agg[t] == 2
            p_11 = Π_gb[1, 1]
            p_00 = Π_gb[2, 2]
        elseif state_space_agg[t - 1] == 2 && state_space_agg[t] == 1
            p_11 = Π_bg[1, 1]
            p_00 = Π_bg[2, 2]
        elseif state_space_agg[t - 1] == 2 && state_space_agg[t] == 2
            p_11 = Π_bb[1, 1]
            p_00 = Π_bb[2, 2]
        end

        for n = 1:N
            shock_idio = rand(dist)
            if state_space_idio[n, t-1] == 1 && shock_idio <= p_11
                state_space_idio[n, t] = 1
            elseif state_space_idio[n, t-1] == 1 && shock_idio > p_11
                state_space_idio[n, t] = 2
            elseif state_space_idio[n, t-1] == 2 && shock_idio <= p_00
                state_space_idio[n, t] = 2
            elseif state_space_idio[n, t-1] == 2 && shock_idio > p_00
                state_space_idio[n, t] = 1
            end
        end
    end

    state_space_agg, state_space_idio
end

##

function get_index(val::Float64, grid::Array{Float64, 1})

    interp =  interpolate(grid, BSpline(Linear()))
    find_index(k) = abs(interp(k) - val)
    index = optimize(find_index, 1.0, length(grid)).minimizer

    index
end

##

function get_prices(prim::Primitives, K_index::Int64, z_index::Int64)
    @unpack α, z_grid, K_grid, L_grid, ē = prim

    r = α * z_grid[z_index] * (K_grid[K_index] / (ē * L_grid[z_index])) ^ (α - 1)
    w = (1 - α) * z_grid[z_index] * (K_grid[K_index] / (ē * L_grid[z_index])) ^ α

    r, w
end

##

function Bellman(prim::Primitives, res::Results)
    @unpack β, δ, nk, k_grid, nε, ε_grid, K_grid, nK, z_grid, nz, Π, ē = prim
    @unpack val_func, a_0, a_1, b_0, b_1 = res

    val_func_next = zeros(nk, nε, nK, nz)
    pol_func_next = zeros(nk, nε, nK, nz)

    interp_k_grid = interpolate(k_grid, BSpline(Linear()))
    interp_val_func = interpolate(val_func, BSpline(Linear()))

    for (K_index, K) in enumerate(K_grid)
        for z_index = 1:nz
            if z_index == 1
                lnKp = a_0 + a_1 * log(K)
            elseif z_index == 2
                lnKp = b_0 + b_1 * log(K)
            end

            Kp = exp(lnKp)
            Kp_index = get_index(Kp, K_grid) #likely not to be an integer
            r, w = get_prices(prim, K_index, z_index)

            for ε_index = 1:nε
                if z_index == 1 && ε_index == 1 #employed & good time
                    row_index = 1
                elseif z_index == 1 && ε_index == 2 #unemployed & good time
                    row_index = 2
                elseif z_index == 2 && ε_index == 1 #employed & bad time
                    row_index = 3
                elseif z_index == 2 && ε_index == 2 #unemployed & bad time
                    row_index = 4
                end

                for (k_index, k) in enumerate(k_grid)
                    budget = r * k + w * ē * ε_grid[ε_index] + (1.0 - δ) * k

                    val_tomorrow(kp_index) = Π[row_index, 1] * interp_val_func(kp_index, 1, Kp_index, 1) + Π[row_index, 2] * interp_val_func(kp_index, 2, Kp_index, 1) + Π[row_index, 3] * interp_val_func(kp_index, 1, Kp_index, 2) + Π[row_index, 4] * interp_val_func(kp_index, 2, Kp_index, 2)

                    val_today(kp_index) = log(budget - interp_k_grid(kp_index)) +  β * val_tomorrow(kp_index)
                    obj(kp_index) = -val_today(kp_index)

                    lower = 1.0
                    upper = get_index(budget, k_grid) # or findlast(kp -> kp < budget, k_grid)

                    kp_index_choice = optimize(obj, lower, upper).minimizer

                    pol_func_next[k_index, ε_index, K_index, z_index] = interp_k_grid(kp_index_choice)
                    val_func_next[k_index, ε_index, K_index, z_index] = val_today(kp_index_choice)
                end
            end
        end
    end

    val_func_next, pol_func_next
end

##

function solve_HH_problem(prim::Primitives, res::Results)
    err = 100.0
    tol = 0.001
    i = 0
    while true
        i += 1
        val_func_next, pol_func_next = Bellman(prim, res)
        err = maximum(abs.(res.val_func - val_func_next))/abs(maximum(val_func_next))
        res.val_func = val_func_next
        res.pol_func = pol_func_next
        # println("\n***** ", i, "th iteration *****")
        # @printf("Absolute difference: %0.4f.\n", float(err))
        # println("***************************")
        if err < tol
            break
        end
    end
    println("\nHH problem is solved after ", i, " iterations.")
end

##

function K_path(prim::Primitives, res::Results)
    @unpack α, β, δ, ē, T, N, K_grid, k_grid, L_grid, p_bg, p_gg, K_ss = prim
    @unpack pol_func, state_space_idio, state_space_agg = res

    K_path_update = zeros(T)
    K_path_update[1] = K_ss
    k_path_update = zeros(N, T)

    for n_index = 1:N
        k_path_update[n_index, 1] = K_path_update[1]
    end

    interp_pol_func = interpolate(res.pol_func, BSpline(Linear()))

    for t_index = 1:(T-1)
        K_index = get_index(K_path_update[t_index], K_grid)
        z_index = state_space_agg[t_index]
        for n_index = 1:N
            ε_index = state_space_idio[n_index, t_index]
            # print(ε_index)
            k_index = get_index(k_path_update[n_index, t_index], k_grid)
            k_path_update[n_index, t_index + 1] = interp_pol_func(k_index, ε_index, K_index, z_index)
        end
        K_path_update[t_index + 1] = mean(k_path_update[:, t_index + 1])
        # @printf("Aggregate K for %0.0f: %0.2f \n", Int(t_index + 1), float(K_path_update[t_index + 1]))
        for int = 1:10
            if t_index == int * 1000
                @printf(" - %0.2f percent is done.\n", float(t_index/11000 * 100))
            end
        end
    end

    K_path_update, k_path_update
end

##

function plot_path(prim::Primitives, res::Results)
    @unpack T, burn, K_ss = prim
    @unpack state_space_agg = res

    X = zeros(T - burn, 2)
    for t_index = 1:(T - burn)
        if state_space_agg[t_index - 1 + burn] == 1
            X[t_index, 1] = res.K_path[t_index - 1 + burn]
        elseif state_space_agg[t_index - 1 + burn] == 2
            X[t_index, 2] = res.K_path[t_index - 1 + burn]
        end
    end
    X1 = X[:, 1]
    X2 = X[:, 2]
    X1 = X1[X1 .!= 0.0]
    X2 = X2[X2 .!= 0.0]
    nX1 = length(X1)
    nX2 = length(X2)
    p1 = plot(collect(1:nX1), [X1 repeat([K_ss], nX1)], labels = ["z = zʰ" "K = Kₛₛ"], title = "Movement of K by z values")
    p2 = plot(collect(1:nX2), [X2 repeat([K_ss], nX2)], labels = ["z = zˡ" "K = Kₛₛ"])
    plt = plot(p1, p2, layout = (2, 1))

    plt
end
##

function OLS(prim::Primitives, res::Results)
    @unpack T, burn = prim
    @unpack state_space_agg, a_0, a_1, b_0, b_1 = res
    # https://juliastats.org/GLM.jl/stable/examples/
    lnK_path = log.(res.K_path)
    Y = lnK_path[burn + 1:T]
    X = zeros(T - burn, 2)

    for t = 1:(T - burn)
        if state_space_agg[t-1+burn] == 1
            X[t, 1] = lnK_path[t - 1 + burn]
        elseif state_space_agg[t - 1 + burn] == 2
            X[t, 2] = lnK_path[t - 1 + burn]
        end
    end

    df1 = DataFrame(x=X[1:T-burn, 1], y=Y[1:T-burn])
    df2 = DataFrame(x=X[1:T-burn, 2], y=Y[1:T-burn])
    df1 = df1[df1."x".!=0, :]
    df2 = df2[df2."x".!=0, :]
    ols1 = lm(@formula(y ~ x), df1)
    ols2 = lm(@formula(y ~ x), df2)

    # reg_err1 = ((coef(ols1)[1] .+ coef(ols1)[2] * df1[:, "x"]) - df1[:, "y"]) ./ df1[:, "y"] * 100
    # sigma1 = std(reg_err1)
    # reg_err2 = ((coef(ols2)[1] .+ coef(ols2)[2] * df2[:, "x"]) - df2[:, "y"])  ./ df2[:, "y"] * 100
    # sigma2 = std(reg_err2)

    ols1, ols2 #sigma1, sigma2
end

##

function solve_model(prim::Primitives, res::Results)
    @unpack burn, nK, T = prim

    error = 100.0
    tol = 0.0005
    λ = 0.5
    i = 0

    res.state_space_agg, res.state_space_idio = draw_shocks(prim, res)
    println("\nShock drawing is done.")
    
    while true
        i += 1
        println("i = ", i)

        println("\nHH problem iteration just started.")
        solve_HH_problem(prim, res)
        println("\nK path calculation just started.")
        res.K_path, res.k_path = K_path(prim, res)
        println("\nK path calculation is done.")

        display(plot_path(prim, res))

        ols1, ols2 = OLS(prim, res)
        println("\nRunning OLS is done.\n")

        a_0_update, a_1_update = coef(ols1)
        b_0_update, b_1_update = coef(ols2)
        R_squared_update = [r2(ols1); r2(ols2)]

        error = abs.(a_0_update - res.a_0)+abs.(a_1_update - res.a_1)+abs.(b_0_update - res.b_0)+abs.(b_1_update - res.b_1)
        error2 = maximum(1.0 .- R_squared_update)
        @printf("a_0_previous: %0.4f, a_0_update: %0.4f \na_1_previous: %0.4f, a_1_update: %0.4f  \nb_0_previous: %0.4f, b_0_update: %0.4f  \nb_1_previous: %0.4f, b_1_update: %0.4f \n", float(res.a_0), float(a_0_update), float(res.a_1), float(a_1_update), float(res.b_0), float(b_0_update), float(res.b_1), float(b_1_update))
        @printf("R²: (%0.4f, %0.4f) \n", float(R_squared_update[1]), float(R_squared_update[2]))

        if error > tol || error2 > tol
            res.a_0 = λ*a_0_update + (1 - λ)*res.a_0
            res.a_1 = λ*a_1_update + (1 - λ)*res.a_1
            res.b_0 = λ*b_0_update + (1 - λ)*res.b_0
            res.b_1 = λ*b_1_update + (1 - λ)*res.b_1
            res.R_squared = R_squared_update
            println("************************************")
            @printf("Error (%0.5f) exceeds the tolerance level (%0.5f).\n", float(error), float(tol))
            println("=> Coefficients are updated.\n\n")
        else
            println("***********************************")
            @printf("Error (%0.5f) is within the tolerance level (%0.5f).\n\n", float(error), float(tol))
            println("The K-S model is solved after ", i, " updates on coffecients! \n")
            println("**** Key Parameters ****")
            @printf("a₀: %0.4f \na₁: %0.4f \nb₀: %0.4f \nb₁: %0.4f \nK_ss: %0.2f", float(res.a_0), float(res.a_1), float(res.b_0), float(res.b_1), float(prim.K_ss))
            @printf("\n\n⇒ If zₜ = zʰ, logK̄ₜ₊₁ = %0.4f + %0.4f × logK̄ₜ with R² = %0.4f \n  If zₜ = zˡ, logK̄ₜ₊₁ = %0.4f + %0.4f × logK̄ₜ with R² = %0.4f.\n", float(res.a_0), float(res.a_1), float(res.R_squared[1]), float(res.b_0), float(res.b_1), float(res.R_squared[2]))
            break
        end
    end
end
