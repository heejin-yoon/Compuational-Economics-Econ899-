##

@with_kw struct Primitives
    β::Float64 = 0.9932 #discount rate
    α::Float64 = 1.5 #risk aversion
    y::Array{Float64, 1} = [1.0, 0.5]
    ny::Int64 = length(y)
    a̲::Float64 = -2.0 #asset lower bound
    ā::Float64 = 5.0 #asset upper bound (ex-post I know the upper bound is less than 1.05)
    na::Int64 = 1000 #number of asset grid points
    a_grid::Array{Float64,1} = collect(range(a̲, length = na, stop = ā)) #capital grid
    Π::Array{Float64,2}=[0.97 0.03; 0.5 0.5] #Markov transition matrix
end

##

mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
    μ::Array{Float64, 2} #distribution of asset holdings and employment
    q::Float64 #price
end

##

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = [zeros(prim.na) zeros(prim.na)] #initial value function guess
    pol_func = [zeros(prim.na) zeros(prim.na)] #initial policy function guess
    μ = [ones(prim.na) ones(prim.na)]./(prim.na*2) #initial μ distribution guess
    q = (prim.β + 1) / 2 ##Price should be between beta and 1, so let's start at the middle point.
    res = Results(val_func, pol_func, μ, q) #initialize results struct
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
    @unpack a_grid, β, na, y, Π, ny = prim

    val_func_next = [zeros(na) zeros(na)]
    pol_func_next = [zeros(na) zeros(na)]

    for y_index = 1:ny
        ap_index_start = 1
        for a_index = 1:na
            val_func_next[a_index, y_index] = -Inf
            for ap_index = ap_index_start:na
                c = y[y_index] + a_grid[a_index] - q*a_grid[ap_index]
                if c > 0
                    U = utility(prim, c)
                    val = U
                    for yp_index = 1:ny
                        val += β * Π[y_index, yp_index] * val_func[ap_index, yp_index]
                    end
                    if val > val_func_next[a_index, y_index]
                        val_func_next[a_index, y_index] = val
                        pol_func_next[a_index, y_index] = a_grid[ap_index]
                        ap_index_start = ap_index
                    end
                end
            end
        end
    end
    val_func_next, pol_func_next
end

##

function Solve_HH_problem(prim::Primitives, res::Results)
    tol = 1e-4
    err = 1.0
    n = 1 # count

    while true
        val_func_next, pol_func_next = T_Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(val_func_next .- res.val_func)) #reset error level
        res.val_func = val_func_next #update value function
        res.pol_func = pol_func_next #update policy function
        # println("***** ", n, "th iteration *****")
        # @printf("Absolute difference: %0.5f \n", float(err))
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
    @unpack pol_func, μ, q = res
    @unpack a_grid, na, y, Π, ny = prim

    μ_next = [zeros(na) zeros(na)]

    for y_index = 1:ny
        for a_index = 1:na
            ap = pol_func[a_index, y_index]
            ap_index = argmin(abs.(ap .- a_grid))
            for yp_index = 1:ny
                μ_next[ap_index, yp_index] += Π[y_index, yp_index] * μ[a_index, y_index]
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
    println("Wealth distribution converged in ", n, " iterations.")
    println(" ")

end

##

function update_q(prim::Primitives, res::Results)
    @unpack a_grid, β = prim
    @unpack μ, q = res

    tol = 1e-3
    n = 1 # count
    excess_demand = sum(μ[:, 1] .* a_grid) + sum(μ[:, 2] .* a_grid)

    if abs(excess_demand) > tol
        if excess_demand < 0
            q_update = q + excess_demand * (q - β)/2
            @printf("Excess demand for borrowing: %0.5f.", float(-excess_demand))
            println(" ")
        elseif excess_demand > 0
            q_update = q + excess_demand * (1 - q)/2
            @printf("Excess demand for bond purchase: %0.5f.", float(excess_demand))
            println(" ")
        end
        println(" ")
        @printf("⇒ Price is adjusted from %0.5f to %0.5f.", float(q), float(q_update))
        println(" ")
        println("****************************")
        res.q = q_update
        return(false)
    else
        @printf("Excess demand: %0.5f.", float(excess_demand))
        println(" ")
        println(" ")
        println("Asset market clears.")
        println(" ")
        @printf("⇒ Price is converged into %0.5f.", float(q))
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
        mkt_clear = update_q(prim, res)
        if mkt_clear == true
            break
        end
    end
end

##

function w_dist(prim::Primitives, res::Results)
    @unpack μ = res
    @unpack a_grid, na, y, ny = prim

    w = [zeros(na) zeros(na)]
    cdf_w = zeros(na)
    cum_w = zeros(na)

    for y_index = 1:ny
        for a_index = 1:na
            wealth = a_grid[a_index] + y[y_index]
            w_index = argmin(abs.(wealth .- a_grid))

            w[w_index, y_index] = μ[a_index, y_index]
        end
    end

    for a_index = 1:na-1
        for y_index = 1:ny
            cdf_w[a_index] += w[a_index, y_index]
            cum_w[a_index] += w[a_index, y_index] * a_grid[a_index]
        end
        cdf_w[a_index+1] = cdf_w[a_index]
        cum_w[a_index+1] = cum_w[a_index]
    end
    for y_index = 1:ny
        cdf_w[na] += w[na, y_index]
        cum_w[na] += w[na, y_index] * a_grid[na]
    end
    cum_w = cum_w ./ maximum(cum_w)

    w, cdf_w, cum_w
end

##

function gini(prim:: Primitives, cdf_w::Array{Float64, 1}, cum_w::Array{Float64, 1})
    @unpack na = prim

    gini = 0.0
    integral = 0.0
    for a_index = 1:na-1
        integral += (cdf_w[a_index+1] - cdf_w[a_index]) * (cum_w[a_index] + cum_w[a_index+1])/2
    end
    gini = (0.5 - integral)/0.5
end

##

function Wfb_lambda(prim::Primitives, res::Results)
    @unpack β, α, Π, y, ny, na = prim
    @unpack val_func, μ = res

    W_fb = 0.0
    λ = [zeros(na) zeros(na)]
    pro_fb = 0.0

    πₑ = Π[2,1] / (1.0 - Π[1,1] + Π[2,1])
    c = πₑ * y[1] + (1 - πₑ) * y[2]
    Util = utility(prim, c)
    W_fb = Util / (1 - β)

    for a_index = 1:na
        for y_index = 1:ny
            λ[a_index, y_index] = ((W_fb + 1/((1 - α) * (1 - β)))/(val_func[a_index, y_index] + + 1/((1 - α) * (1 - β))))^(1/(1 - α)) - 1
        end
    end
    I_pro_fb = (λ .>= 0)
    pro_fb = sum(I_pro_fb .* μ)

    W_fb, λ, pro_fb
end

##

function W_inc(prim::Primitives, res::Results)
    @unpack β, α, Π, y, ny, na = prim
    @unpack val_func, μ = res

    W_inc = 0.0

    for a_index = 1:na
        for y_index = 1:ny
            W_inc += μ[a_index, y_index] * val_func[a_index, y_index]
        end
    end
    W_inc
end
