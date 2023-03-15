##

@with_kw struct Primitives
    β::Float64 = 0.99
    α::Float64 = 0.36
    k̲::Float64 = 0.0
    k̄::Float64 = 0.5
    nk::Int64 = 20
    k_grid::Array{Float64, 1} = collect(range(k̄, length = nk, stop = k̲))
    K̲::Float64 = 0.15
    K̄::Float64 = 0.25
    nK::Int64 = 10
    K_grid::Array{Float64, 1} = collect(range(K̲, length = nK, stop = K̄))
    w::Array{Float64, 1} = (1 - α) .* K_grid .^ α
    r::Array{Float64, 1} = α .* K_grid .^ (α - 1)
end

##

mutable struct Results
    val_func::Array{Float64, 2}
    pol_func::Array{Float64, 2}
end

##

function Initialize()
    prim = Primitives()
    value_func = zeros(prim.nk, prim.nK)
    pol_func = zeros(prim.nk, prim.nK)
    res = Results(value_func, pol_func)
    prim, res
end

##

function Bellman(prim::Primitives, res::Results, degree::String)
    @unpack k_grid, K_grid, α, β, nk, nK, w, r = prim
    @unpack val_func = res

    val_func_next = zeros(nk, nK)
    pol_func_next = zeros(nk, nK)

    interp_k_grid = interpolate(k_grid, BSpline(Linear()))

    if degree == "Linear"
        interp_val_func = interpolate(val_func, BSpline(Linear()))
    elseif degree == "Cubic"
        interp_val_func = interpolate(val_func, (BSpline(Cubic(Line(OnGrid()))), BSpline(Linear())))
    end

    for K_index = 1:nK, k_index = 1:nk
        w = (1 - α) * K_grid[K_index] ^ α
        r = α * K_grid[K_index] ^ (α - 1)
        val(kp) = log(r * k_grid[k_index] + w - interp_k_grid(kp)) + β * interp_val_func(kp, K_index)
        obj(kp) = -val(kp)

        lower = 1.0
        upper = findlast(kp -> kp < r * k_grid[k_index] + w, interp_k_grid)

        kp_choice = optimize(obj, lower, upper).minimizer
        val_func_next[k_index, K_index] = val(kp_choice)
        pol_func_next[k_index, K_index] = interp_k_grid(kp_choice)
    end
    val_func_next, pol_func_next
end

##

function solve_model(prim::Primitives, res::Results, degree::String)

    tol = 0.001
    err = 100.0

    i = 1

    while true
        val_func_next, pol_func_next = Bellman(prim, res, degree)
        err = maximum(abs.(val_func_next .- res.val_func))
        res.val_func = val_func_next
        res.pol_func = pol_func_next
        println("***** ", i, "th iteration *****")
        @printf("Absolute difference: %0.4f > %0.4f", float(err), float(tol))
        println("")
        println("***************************")
        i += 1
        if err < tol #begin iteration
            break
        end
    end
    @printf("Absolute difference: %0.4f ≤ %0.4f", float(err), float(tol))
    println(" ")
    @printf("Value function iteration is done after %0.0f iterations.", Int(i))
    println(" ")

    res
end

##

function ss_index(prim::Primitives, res::Results)
    @unpack k_grid, K_grid, β, α, nk, nK = prim

    k_ss = (α*β)^(1/(1-α))

    interp_k =  interpolate(k_grid, BSpline(Linear()))
    interp_K =  interpolate(K_grid, BSpline(Linear()))
    find_k(k) = abs(interp_k(k) - k_ss)
    find_K(K) = abs(interp_K(K) - k_ss)
    k_ss_index = optimize(find_k, 1.0, nk).minimizer
    K_ss_index = optimize(find_K, 1.0, nK).minimizer
    k_ss_index, K_ss_index
end
