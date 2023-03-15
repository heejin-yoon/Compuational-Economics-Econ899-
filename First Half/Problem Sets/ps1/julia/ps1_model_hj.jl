
## keyword-enabled structure to hold model primitives

@with_kw struct Primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 75.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(k_min, length = nk, stop = k_max)) #capital grid
    ζ::Array{Float64,1}=[1.25, 0.2]
    Π::Array{Float64,2}=[0.977 0.023; 0.074 0.926]
    nz::Int64 = length(ζ) #number of states
end

## structure that holds model results

mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
end

## function for initializing model primitives and results

function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = [zeros(prim.nk) zeros(prim.nk)] #initial value function guess
    pol_func = [zeros(prim.nk) zeros(prim.nk)] #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

## Bellman Operator

function Bellman(prim::Primitives, res::Results)
    @unpack val_func = res
    @unpack k_grid, β, δ, α, nk, ζ, Π, nz = prim

    val_func_cand = [zeros(nk) zeros(nk)]
    pol_func_cand = [zeros(nk) zeros(nk)]

    for z_index = 1:nz
        kp_index_start = 1
        for k_index = 1:nk
            y = ζ[z_index]*(k_grid[k_index])^α
            val_func_cand[k_index, z_index] = -Inf
            for kp_index = kp_index_start:nk
                c = y + (1 - δ) * k_grid[k_index] - k_grid[kp_index]
                if c > 0
                    util = log(c)
                    val = util
                    for zp_index = 1:nz
                        val += β * Π[z_index, zp_index] * val_func[kp_index, zp_index]
                    end
                    if val > val_func_cand[k_index, z_index]
                        val_func_cand[k_index, z_index] = val
                        pol_func_cand[k_index, z_index] = k_grid[kp_index]
                        kp_index_start = kp_index
                    end
                end
            end
        end
    end
    val_func_cand, pol_func_cand
end

## Value function iteration

function Solve_model(prim::Primitives, res::Results)

    tol = 1e-4
    err = 1.0
    n = 1 # count

    while true #begin iteration
        val_func_cand, pol_func_cand = Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(val_func_cand .- res.val_func)) #reset error level
        res.val_func = val_func_cand #update value function
        res.pol_func = pol_func_cand
        println("********* ", n, "th iteration *********")
        @printf("Absolute difference: %0.5f", float(err))
        println("")
        println("***********************************")
        println("")
        n+=1
        if err < tol
            println("********* ", n, "th iteration *********")
            @printf("Absolute difference: %0.5f", float(err))
            println("")
            println("***********************************")
            println("")
            break
        end
    end
    println("Value function converged in ", n, " iterations.")
    println("")

end
