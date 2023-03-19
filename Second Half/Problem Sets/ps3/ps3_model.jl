## keyword-enabled structure to hold model primitives


@with_kw struct Primitives
    data1 = sort!(DataFrame(load(rt * "/Second Half/Problem Sets/ps3/Car_demand_characteristics_spec1.dta")), [:Year, :Model_id])
    data2 = sort!(DataFrame(load(rt * "/Second Half/Problem Sets/ps3/Car_demand_iv_spec1.dta")), [:Year, :Model_id])
    data3 = DataFrame(load(rt * "/Second Half/Problem Sets/ps3/Simulated_type_distribution.dta"))
    markets = sort!(unique(data1.Year))
end


mutable struct Results
    β::Vector{Float64}
    λ::Float64
    δ::Vector{Float64}
end


function Initialize()
    prim = Primitives()
    β = zeros(47)
    λ = 0.6
    δ = zeros(size(prim.data1, 1))
    res = Results(β, λ, δ)
    prim, res
end


function market_share(data_market, δ::Array{Float64}, λ::Float64)

    X = select(data_market, :price, :dpm, :hp2wt, :size, :turbo, :trans, :Year_1986, :Year_1987, :Year_1988, :Year_1989, :Year_1990, :Year_1991, :Year_1992, :Year_1993, :Year_1994, :Year_1995, :Year_1996, :Year_1997, :Year_1998, :Year_1999, :Year_2000, :Year_2001, :Year_2002, :Year_2003, :Year_2004, :Year_2005, :Year_2006, :Year_2007, :Year_2008, :Year_2009, :Year_2010, :Year_2011, :Year_2012, :Year_2013, :Year_2014, :Year_2015, :model_class_2, :model_class_3, :model_class_4, :model_class_5, :cyl_2, :cyl_4, :cyl_6, :cyl_8, :drive_2, :drive_3, :Intercept)
    J = size(X, 1)
    price = Vector{Float64}(X[:, :price])

    y = Array{Float64}(prim.data3)

    s_ind = zeros(J, 100)
    s_mkt = zeros(J)

    for i_index = 1:100
        μ = λ * y[i_index] * price
        exp_δμ = @. exp(δ + μ)

        denom = 1 + sum(exp_δμ)
        s_ind[:, i_index] = exp_δμ / denom

    end

    s_mkt = mean(s_ind, dims=2)

    s_mkt, s_ind
end


function get_δ(market, λ::Float64, method)

    data_market = prim.data1[prim.data1.Year.==market, :]

    δ = Vector{Float64}(data_market[:, :delta_iia])
    s_jt = Vector{Float64}(data_market[:, :share])
    J = size(δ, 1)
    J = size(δ, 1)

    tol = 1e-12
    i = 0
    difff = 0.0

    if method == "Contraction"

        while true
            i += 1
            println(" ")
            println("************* Trial #", i, " *************")

            s_new = market_share(data_market, δ, λ)[1]
            difff = sum(abs.(log.(s_jt) - log.(s_new)))

            if difff > tol
                println("")
                @printf("The difference is above the tolerance level: %0.12f.", float(difff))
                println(" ")
                δ_new = δ + log.(s_jt) - log.(s_new)
                δ = δ_new
                println(" ")
                println("************************************")
            else
                break
            end
        end

    elseif method == "Newton"

        s_0 = 1 - sum(s_jt)
        δ = log.(s_jt) .- log(s_0)

        while true
            i += 1
            # println(" ")
            # println("************* Trial #", i, " *************")

            s_mkt_new = market_share(data_market, δ, λ)[1]
            s_ind_new = market_share(data_market, δ, λ)[2]
            NS = size(s_ind_new, 2)
            Δ_δ = -(1 / NS * (Matrix(1.0I, J, J) .* (s_ind_new * (1 .- s_ind_new)') - (1 .- Matrix(1.0I, J, J)) .* (s_ind_new * s_ind_new'))) ./ s_mkt_new
            difff = sum(abs.(log.(s_jt) - log.(s_mkt_new)))

            if difff > 1.0
                # println("")
                # @printf("The difference is above the tolerance level: %0.12f.", float(difff))
                # println(" ")
                δ_new = δ + log.(s_jt) - log.(s_mkt_new)
                δ = δ_new
                # println(" ")
                # println("************************************")

            elseif difff > tol && difff <= 1
                # println("")
                # @printf("The difference is above the tolerance level: %0.12f.", float(difff))
                # println(" ")
                δ_new = δ - (inv(Δ_δ) * (log.(s_jt) - log.(s_mkt_new)))
                δ = δ_new
                # println(" ")
                # println("************************************")

            else
                break
            end
        end
    
    else
        println("Please specify a correct method of demand inversion.")
        δ = "Error"
    end

    return δ
end


function object_function(λ::Float64, method, gmm_method)

    M = size(prim.markets, 1)
    
    δ = []  
    @threads for M_index = 1:M
        δ_new = get_δ(prim.markets[M_index], λ, method)
        δ = vcat(δ, δ_new)
    end

    # show(stdout, propertynames(iv_market))

    X = Array{Float64}(select(prim.data1, :price, :dpm, :hp2wt, :size, :turbo, :trans, :Year_1986, :Year_1987, :Year_1988, :Year_1989, :Year_1990, :Year_1991, :Year_1992, :Year_1993, :Year_1994, :Year_1995, :Year_1996, :Year_1997, :Year_1998, :Year_1999, :Year_2000, :Year_2001, :Year_2002, :Year_2003, :Year_2004, :Year_2005, :Year_2006, :Year_2007, :Year_2008, :Year_2009, :Year_2010, :Year_2011, :Year_2012, :Year_2013, :Year_2014, :Year_2015, :model_class_2, :model_class_3, :model_class_4, :model_class_5, :cyl_2, :cyl_4, :cyl_6, :cyl_8, :drive_2, :drive_3, :Intercept))
    IV = Array{Float64}(select(prim.data2, :i_import, :diffiv_local_0, :diffiv_local_1, :diffiv_local_2, :diffiv_local_3, :diffiv_ed_0))
    Z = [X[:, 2:end] IV]

    β = ((X' * Z * inv(Z' * Z) * Z' * X)^(-1)) * (X' * Z * inv(Z' * Z) * Z' * δ)
    ρ = δ - X * β


    if gmm_method == "1step"

        ObjFn = (ρ'*Z*inv(Z'*Z)*Z'*ρ)[1, 1]
    
    elseif gmm_method == "2step"

        S = (Z .* ρ)' * (Z .* ρ)
        β = ((X' * Z * inv(S) * Z' * X)^(-1)) * (X' * Z * inv(S) * Z' * δ)
        ρ = δ - X * β

        res.β = vec(β)

        ObjFn = (ρ'*Z*inv(S)*Z'*ρ)[1, 1]

    else
        println("Please specify a correct GMM method.")
        ObjFn = 1e100
    end

    return ObjFn
end


function solve_GMM(method, gmm_method)

    optim = optimize(lambda -> object_function(lambda, method, gmm_method), 0.0, 1.0, Brent(); show_trace = true)

    λ = optim.minimizer
    minval = optim.minimum

    return λ, minval
end
