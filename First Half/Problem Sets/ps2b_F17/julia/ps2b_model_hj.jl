using Parameters, Plots, Printf, Weave

##

@with_kw struct Primitives
    β::Float64 = 0.8 #discount rate
    α::Float64 = 1.5 #risk aversion
    y::Array{Float64, 1} = [1.0, 0.05]
    ny::Int64 = length(y)
    a̲::Float64 = -0.525 #borrowing constraint
    ā::Float64 = 4.0 #asset upper bound (ex-post I know the upper bound is less than 1.05)
    a_inc::Float64 = 0.001 #increment of a_grid
    a_grid::Array{Float64,1} = collect(a̲:a_inc:ā) #capital grid
    na::Int64 = length(a_grid) #number of asset grid points
    Π::Array{Float64,2}=[0.75 0.25; 0.25 0.75]
    r::Float64 = 0.04 # real interest rate
    ρ::Float64 = 0.9 #legal record keeping technology
    a_index_zero::Int64 = findall(x -> x == 0, a_grid)[1] #locaton of zero in asset grid
end

##

mutable struct Results
    val_func::Array{Float64, 3} #value function
    pol_func::Array{Float64, 3} #policy function
    def_func::Array{Float64, 2} #policy function
    μ::Array{Float64, 3} #distribution of asset holdings and employment
    q::Array{Float64, 2} #price
end

##

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.na, prim.ny, 2) #initial value function guess
    pol_func = zeros(prim.na, prim.ny, 2) #initial policy function guess
    def_func = zeros(prim.na, prim.ny) #initial default function guess
    μ = ones(prim.na, prim.ny, 2)./(prim.na*prim.ny*2) #initial policy function guess
    q = (1 / (1 + prim.r)) .* ones(prim.na, prim.ny)
    res = Results(val_func, pol_func, def_func, μ, q) #initialize results struct
    prim, res #return deliverables
end

##

function utility(prim::Primitives, c::Float64)
    @unpack α = prim

    util = (c^(1 - α) - 1)/(1 - α)
end

##

function T_Bellman(prim::Primitives, res::Results)
    @unpack val_func, q = res
    @unpack a_grid, β, na, y, Π, ny, a_index_zero, ρ, r = prim

    val_func_next = zeros(na, ny, 2)
    pol_func_next = zeros(na, ny, 2)
    def_func_next = zeros(na, ny)

        # nondefaulted state (h = 0)
        for y_index = 1:ny
            ap_index_start = 1

            val_def = utility(prim, y[y_index])

            for yp_index = 1:ny
                val_def += β * Π[y_index, yp_index] * val_func[a_index_zero, yp_index, 2]
            end

            for a_index = 1:na
                val_func_next[a_index, y_index, 1] = -Inf

                for ap_index = ap_index_start:na
                    c = y[y_index] + a_grid[a_index] - q[ap_index, y_index] * a_grid[ap_index]

                    if c > 0
                        val = utility(prim, c)
                        for yp_index = 1:ny
                            val += β * Π[y_index, yp_index] * val_func[ap_index, yp_index, 1]
                        end
                        if val > val_func_next[a_index, y_index, 1]
                            val_func_next[a_index, y_index, 1] = val
                            pol_func_next[a_index, y_index, 1] = a_grid[ap_index]
                            ap_index_start = ap_index
                        end
                    end
                end
                if val_func_next[a_index, y_index, 1] >= val_def
                    def_func_next[a_index, y_index] = 0

                elseif val_def > val_func_next[a_index, y_index, 1]
                    def_func_next[a_index, y_index] = 1
                    pol_func_next[a_index, y_index, 1] = 0
                    val_func_next[a_index, y_index, 1] = val_def
                end
            end
        end

        # defaulted state (h = 1)
        for y_index = 1:ny
            ap_index_start = a_index_zero

            for a_index = 1:na
                val_func_next[a_index, y_index, 2] = -Inf

                for ap_index = ap_index_start:na
                    c = y[y_index] + a_grid[a_index] - 1/(1+r) * a_grid[ap_index]

                    if c > 0
                        val = utility(prim, c)
                        for yp_index = 1:ny
                            val += β * Π[y_index, yp_index] * (ρ * val_func[ap_index, yp_index, 2] + (1 - ρ) * val_func[ap_index, yp_index, 1])
                        end
                        if val > val_func_next[a_index, y_index, 2]
                            val_func_next[a_index, y_index, 2] = val
                            pol_func_next[a_index, y_index, 2] = a_grid[ap_index]
                            ap_index_start = ap_index
                        end
                    end
                end
            end
        end
    val_func_next, pol_func_next, def_func_next
end


##

function Solve_HH_problem(prim::Primitives, res::Results)
    tol = 1e-4
    err = 1.0
    n = 1 # count

    while true
        val_func_next, pol_func_next, def_func_next = T_Bellman(prim, res) #spit out new vectors
        abs_diff = abs.(val_func_next .- res.val_func)
        replace!(abs_diff, NaN=>0)
        err = maximum(abs_diff) #reset error level
        res.val_func = val_func_next #update value function
        res.pol_func = pol_func_next #update policy function
        res.def_func = def_func_next
        # println("***** ", n, "th iteration *****")
        # println("Absolute difference: ", err)
        # println("***************************")
        n += 1
        if err < tol #begin iteration
            break
        end
    end
    println("HH Value function converged in ", n, " iterations.")
    println(" ")

end

##

function Tstar_Bellman(prim::Primitives, res::Results)
    @unpack pol_func, def_func, μ, q = res
    @unpack a_grid, na, y, Π, ny, ρ, a_index_zero = prim

    μ_next = zeros(prim.na, prim.ny, 2)
    check = zeros(prim.na, prim.ny, 2)
    ap_index = 0

    for y_index = 1:ny
        for a_index = 1:na
            for h_index = 1:2
                ap = pol_func[a_index, y_index, h_index]
                ap_index = argmin(abs.(ap .- a_grid))
                # check[a_index, y_index, h] = def_func[a_index, y_index] * (2 - h)
                if h_index == 1
                    if def_func[a_index, y_index] == 1
                        for yp_index = 1:ny
                            μ_next[a_index_zero, yp_index, 2] += Π[y_index, yp_index] * μ[a_index, y_index, h_index]
                        end
                    elseif def_func[a_index, y_index] == 0
                        for yp_index = 1:ny
                            μ_next[ap_index, yp_index, 1] += Π[y_index, yp_index] * μ[a_index, y_index, h_index]
                        end
                    end
                elseif h_index == 2
                    for yp_index = 1:ny
                        μ_next[ap_index, yp_index, 1] += Π[y_index, yp_index] * μ[a_index, y_index, h_index] * (1 - ρ)
                        μ_next[ap_index, yp_index, 2] += Π[y_index, yp_index] * μ[a_index, y_index, h_index] * ρ
                    end
                end
            end
        end
    end
    μ_next
end

##

function Solve_μ_dist(prim::Primitives, res::Results)
    tol = 1e-4
    err = 1.0
    n = 1 # count

    while true
        μ_next = Tstar_Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(μ_next .- res.μ)) #reset error level
        res.μ = μ_next #update μ distribution
        # println("***** ", n, "th iteration *****")
        # println("Absolute difference: ", err)
        # println("***************************")
        n+=1
        if err < tol #begin iteration
            break
        end
    end
    println("μ distribution converged in ", n, " iterations.")
    println(" ")

end

##

function update_q(prim::Primitives, res::Results)
    @unpack a_grid, β, ny, r, na, Π, a_index_zero = prim
    @unpack μ, pol_func, def_func, q = res

    tol = 1e-3
    err = 1.0
    # n = 1

    tot_borrow = 0.0
    tot_loss = 0.0
    loss_rate = 0.0
    ap_index = 0

    for y_index = 1:ny
        for a_index = 1:na
            ap = pol_func[a_index, y_index, 1]
            if ap < 0
                tot_borrow += μ[a_index, y_index, 1] * ap
                ap_index = argmin(abs.(ap .- a_grid))
                for yp_index = 1:ny
                    tot_loss += μ[a_index, y_index, 1] * ap * Π[y_index, yp_index] * def_func[ap_index, yp_index]
                end
            end
        end
    end
    loss_rate = tot_loss/tot_borrow
    err = q[1, 1] - (1 - loss_rate) / (1 + r)
    if abs(err) > tol
        q_update = [(q[1:(a_index_zero-1), :] .- err * 0.5); q[a_index_zero:na, :]]
        @printf("Price difference: %0.5f.", float(err))
        println(" ")
        println(" ")
        @printf("⇒ Borrowing price is adjusted from %0.5f to %0.5f.", float(q[1, 1]), float(q_update[1, 1]))
        println(" ")
        println("****************************")
        res.q = q_update
        return(false)
    else
        @printf("Price difference: %0.5f.", float(err))
        println(" ")
        println(" ")
        @printf("⇒ Borrowing price is converged into %0.5f.", float(q[1, 1]))
        println(" ")
        println("****************************")
        return(true)
    end
end

##

function Solve_model(prim::Primitives, res::Results)
    n = 0
    while true
        n += 1
        println(" ")
        println("********* Trial #", n, " *********")

        Solve_HH_problem(prim, res)
        Solve_μ_dist(prim, res)
        equilibrium = update_q(prim, res)
        if equilibrium == true
            break
        end
    end
end

##
