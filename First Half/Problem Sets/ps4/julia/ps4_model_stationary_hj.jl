## keyword-enabled structure to hold model Primitives

@with_kw struct Primitives                                                       # define parameters
    N::Int64 = 66                                                                # lifespan
    n::Float64 = 0.011                                                           # population growth rate
    J_retire::Int64 = 46
    θ::Float64 = 0.11                                                            #labor income tax                                                            # labor income tax rate
    σ::Float64 = 2.0                                                             # relative risk aversion coefficient
    γ::Float64 = 0.42                                                            # weight on consumption in utility function
    η::Matrix{Float64} = readdlm("ef.txt", '\t', Float64, '\n')
    z::Array{Float64, 1} = [3.0, 0.5]                                            # idiosyncratic productivity
    e::Matrix{Float64} = η*z'                                                    # worker's productivity (z*η: 45x2 matrix)
    nz::Int64 = length(z)                                                       # number of productivity states
    Π₀::Array{Float64, 1} = [0.2037, 0.7963]                                     # idiosyncratic productivity
    Π::Matrix{Float64} = [0.9261 1-0.9261;1-0.9811 0.9811]                       # persistence probabilities [πₕₕ πₕₗ; πₗₕ πₗₗ]
    α::Float64 = 0.36                                                            # capital share
    δ::Float64 = 0.06                                                            # depreciation rate
    β::Float64 = 0.97                                                            # discount rate
    a̲::Int64 = 0.0                                                            # lower asset boundary
    ā::Int64 = 75.0                                                           # upper asset boundary
    na::Int64 = 500                                                            # number of asset grids
    a_grid::Array{Float64, 1} = collect(range(a̲, stop = ā, length = na))  # asset grid array
end

## structure that holds model results

mutable struct Results
    val_func::Array{Float64, 3}                                                     # value function
    pol_func::Array{Float64, 3}                                                     # policy function
    lab_func::Array{Float64, 3}                                                     # labor function
    K::Float64                                                                  # aggregate capital
    L::Float64                                                                  # aggregate labor
    F::Array{Float64, 3}                                                            # asset distribution
    w::Float64                                                                   # wage
    r::Float64                                                                   # interest rate
    b::Float64                                                                   # pension benefit
    W::Float64                                                                   # welfare
    CV::Float64                                                                  # CV
end

## function for initializing model primitives and results

function Initialize()                                                            # initialize primtives
    prim = Primitives()
    val_func = zeros(prim.N, prim.na, prim.nz)                                                # initial val_func guess: zero
    pol_func = zeros(prim.N, prim.na, prim.nz)                                                # initial pol_func guess: zero
    lab_func = zeros(prim.N, prim.na, prim.nz)                                                # initial lab_func guess: zero
    K = 3.3                                                                     # initial guess on aggregate capital
    L = 0.3                                                                     # initial guess on aggregate labor
    F = zeros(prim.N, prim.na, prim.nz)                                                       # initialize asset distribution
    w = 1.05                                                                     # initial assumption on w
    r = 0.05                                                                     # initial assumption on r
    b = 0.2                                                                      # initial assumption on b
    W = 0.0
    CV = 0.0
    res = Results(val_func, pol_func, lab_func, K, L, F, w, r, b, W, CV)      # initialize results struct
    prim = Primitives()
    prim, res
end

## utility function of retired people

function utility(prim::Primitives, res::Results, c::Float64, l::Float64, J::Int64)
    @unpack σ, γ, J_retire = prim

    u = -Inf
    if J >= J_retire
        if c>0
            u = (c^((1-σ) * γ))/(1 - σ)
        end
    elseif J < J_retire
        if (c>0)&&(l>=0)&&(l<=1)
            u = (((c^γ) * ((1 - l)^(1-γ)))^(1-σ))/(1 - σ)
        end
    end

    u
end

## optimal labor choice

function optimal_labor(prim::Primitives, res::Results, a::Float64, ap::Float64, j::Int64, z_index::Int64)
    @unpack γ, θ, e, η = prim
    @unpack w, r = res

    l::Float64 = (γ*(1-θ)*e[j,z_index]*w - (1-γ)*((1+r)*a - ap)) / ((1-θ)*w*e[j,z_index])
    if l>1
        l = 1
    elseif l<0
        l = 0
    end
    l
end

## Bellman equation

function Bellman(prim::Primitives, res::Results)
    @unpack N, J_retire, a_grid, na, β, nz, θ, e, Π = prim
    @unpack r, b, w = res

    val_func = zeros(N, na, nz)
    pol_func = zeros(N, na, nz)
    lab_func = zeros(N, na, nz)

    l = 0.0
    for a_index = 1:na
        c = (1+r)*a_grid[a_index]+b
        lab_func[N, a_index, 1] = l
        val_func[N, a_index, 1] = utility(prim, res, c, l, N)
    end

    for j = N-1:-1:J_retire
        for a_index = 1:na
            val_func[j, a_index, 1] = -Inf
            lab_func[j, a_index, 1] = l
            for ap_index=1:na
                c = (1 + r) * a_grid[a_index] + b - a_grid[ap_index]
                val = utility(prim, res, c, l, j) + β*val_func[j+1, ap_index, 1]
                if val > val_func[j, a_index, 1]
                    val_func[j, a_index, 1] = val
                    pol_func[j, a_index, 1] = a_grid[ap_index]
                end
            end
        end
    end
    val_func[:, :, 2] = val_func[:, :, 1]
    pol_func[:, :, 2] = pol_func[:, :, 1]
    lab_func[:, :, 2] = lab_func[:, :, 1]

    for j = (J_retire - 1):-1:1
        for a_index = 1:na, z_index = 1:nz
            val_func[j, a_index, z_index] = -Inf
            for ap_index=1:na
                l = optimal_labor(prim, res, a_grid[a_index], a_grid[ap_index], j, z_index)
                c = w * (1 - θ) * e[j, z_index] * l + (1 + r) * a_grid[a_index] - a_grid[ap_index]
                val = utility(prim, res, c, l, j)
                for zp_index = 1:nz
                    val += β * Π[z_index, zp_index] * val_func[j+1, ap_index, zp_index]
                end
                if val > val_func[j, a_index, z_index]
                    val_func[j, a_index, z_index] = val
                    pol_func[j, a_index, z_index] = a_grid[ap_index]
                    lab_func[j, a_index, z_index] = l
                end
            end
        end
    end
    val_func, pol_func, lab_func
end

## distribution of population by age and asset grid

function F_dist(prim::Primitives, res::Results)
    @unpack a_grid, N, n, na, nz, Π₀, Π = prim
    @unpack pol_func = res

    F = zeros(N, na, nz)                                          # initialize again
    F[1, 1, :] = Π₀                                                          # j=1, a=0, z=zᴴ or zᴸ
    weight = ones(N)
    μ = ones(N)

    for j = 1:(N-1)
        for a_index = 1:na, z_index = 1:nz
            if F[j, a_index, z_index] > 0
                ap = pol_func[j, a_index, z_index]
                ap_index = argmin(abs.(ap .- a_grid))
                for zp_index = 1:nz
                    F[j+1, ap_index, zp_index] += Π[z_index, zp_index]*F[j, a_index, z_index]
                end
            end
        end
    end

    for i = 1:(N-1)
        weight[i+1] = weight[i]/(1+n)
    end

    μ = weight./sum(weight)

    for z_index=1:nz
        F[:, :, z_index]= F[:, :, z_index] .* μ
    end
    F
end

## aggregate economy-wide labor

function aggregate_L(prim::Primitives, res::Results)
    @unpack na, nz, N, e, a_grid = prim
    @unpack F, lab_func = res

    E_grid = zeros(N, na, nz)
    for a_index = 1:na, z_index = 1:nz
        E_grid[1:45, a_index, z_index] = e[:, z_index]
    end
    L = sum(F .* lab_func .* E_grid)

    L
end

## aggregate economy-wide capital

function aggregate_K(prim::Primitives, res::Results)
    @unpack na, nz, N, e, a_grid = prim
    @unpack F = res

    A_grid = zeros(N, na, nz)
    for j = 1:N, z_index = 1:nz
        A_grid[j, :, z_index] = a_grid
    end
    K = sum(F .* A_grid)

    K
end

## update prices

function update_prices(prim::Primitives, res::Results)
    @unpack α, δ, J_retire, N, θ = prim
    @unpack K, L, w, r, b, F = res

    w = (1 - α) * (K ^ α) * (L ^ (- α))
    r = α * (K ^ (α - 1)) * (L ^ (1-α)) - δ
    b = (θ * w * L) / sum(F[J_retire:N, :, :])

    w, r, b
end

## Solve the Model

function solve_model(prim::Primitives, res::Results)

    tol = 0.01
    i = 0
    λ = 0.5
    K1 = 0.0
    L1 = 0.0

    while true
        i += 1
        println(" ")
        println("************* Trial #", i, " *************")

        res.val_func, res.pol_func, res.lab_func = Bellman(prim, res)
        println("HH problem is solved.")

        res.F = F_dist(prim, res)
        println("F distribution is calculated.")

        K1 = aggregate_K(prim, res)
        L1 = aggregate_L(prim, res)
        diff = abs(res.K - K1)+abs(res.L - L1)
        println("Aggregate L&K calculation is done.")
        println("")
        @printf("Capital demand: %0.3f.", float(res.K))
        println(" ")
        @printf("Labor demand: %0.3f.", float(res.L))
        println(" ")
        @printf("Capital supply: %0.3f.", float(K1))
        println(" ")
        @printf("Labor supply: %0.3f.", float(L1))
        println(" ")

        @printf("Difference: %0.3f.", float(diff))
        println(" ")

        if diff > tol
            K_update = λ * K1 + (1 - λ) * res.K
            L_update = λ * L1 + (1 - λ) * res.L
            println("")
            @printf("- Aggregate K is adjusted from %0.3f to %0.3f.", float(res.K), float(K_update))
            println(" ")
            @printf("- Aggregate L is adjusted from %0.3f to %0.3f.", float(res.L), float(L_update))
            println(" ")
            res.K = K_update
            res.L = L_update
            res.w, res.r, res.b = update_prices(prim, res)
            println(" ")
            println("************************************")
        else
            break
        end
    end
    println("***************************************")
    println(" ")
    println("★ Market Clearing Outcome ★  ")
    println(" ")

    @printf("Number of iterations: %0.0f.", i)
    println(" ")
    @printf("Aggregate Capital: %0.3f.", res.K)
    println(" ")
    @printf("Aggregate Labor: %0.3f.", res.L)
    println(" ")
    @printf("Wage: %0.3f.", res.w)
    println(" ")
    @printf("Interest rate: %0.3f.", res.r)
    println(" ")
    @printf("Social security benefits: %0.3f.", res.b)
    println(" ")
    println(" ")
    println("***************************************")
end
#

## Welfare Calculation

function welfare_analysis(prim::Primitives, res::Results)
    @unpack a_grid, na, nz, N = prim

    A_grid = zeros(N, na, nz)
    welfare = res.val_func .* res.F
    res.W = sum(welfare[isfinite.(welfare)])


    for j=1:N, z_index=1:nz
        A_grid[j, :, z_index] = a_grid
    end

    wealth_first_moment = sum(res.F.* A_grid)
    wealth_second_moment = sum(res.F.* (A_grid.^2))
    wealth_variance = wealth_second_moment - (wealth_first_moment^2)
    res.CV = wealth_first_moment / sqrt(wealth_variance)

    res
end
