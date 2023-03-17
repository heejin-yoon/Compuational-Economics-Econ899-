@with_kw struct Primitives
    data = DataFrame(load(rt * "/Second Half/Problem Sets/ps2/Mortgage_performance_data.dta"))
    KPU_1d::Array{Float64} = Array(DataFrame(CSV.File(rt * "/Second Half/Problem Sets/ps2/KPU_d1_l20.csv")))
    KPU_2d::Array{Float64} = Array(DataFrame(CSV.File(rt * "/Second Half/Problem Sets/ps2/KPU_d2_l20.csv")))
    θ_initial::Array{Float64} = [0.0; -1.0; -1.0; Array(fill(0.0, 15)); 0.3; 0.5]

    # θ_initial::Array{Float64} = [-5.48100226949492
    #     -2.614259560073641
    #     -2.2323474393428664
    #     0.40452302167241466
    #     0.27924492325612493
    #     0.264782325756695
    #     0.06258457636401359
    #     0.15085958657513318
    #     -0.04698336957419711
    #     0.10285115237450823
    #     0.4268824649599777
    #     0.21712408213320744
    #     -0.18340344234877518
    #     0.30116878763758176
    #     0.5115433213163416
    #     0.1339203500571433
    #     -0.0703953500654598
    #     -0.07471452242530689
    #     0.08134580158999291
    #     0.29460879975537024] ## result derived from quadrature

    # θ_initial::Array{Float64} = [-3.9723172500266224
    # -2.402582852785967
    # -2.1782638391760294
    # 0.36355234144982795
    # 0.25026179176679386
    # 0.23873990962319872
    # 0.059803374621062146
    # 0.15850324945992733
    # -0.04600426928602499
    # 0.09609348616549379
    # 0.38673547440297484
    # 0.18989849776614687
    # -0.16169195894614594
    # 0.2732567337013563
    # 0.4528667473216782
    # 0.1260416127719544
    # -0.045534668974340484
    # -0.05540421765744506
    # 0.07623536830210893
    # 0.15017309288787165] ## result derived from ghk
end


mutable struct Results
    θ_quadrature::Array{Float64}
    θ_ghk::Array{Float64}
    θ_acceptreject::Array{Float64}
    L::Array{Float64}
end


function Initialize()
    prim = Primitives()
    θ_quadrature = zeros(20)
    θ_ghk = zeros(20)
    θ_acceptreject = zeros(20)
    L_quadrature = zeros(size(prim.data, 1))
    L_ghk = zeros(size(prim.data, 1))
    L_acceptreject = zeros(size(prim.data, 1))
    res = Results(θ_quadrature, θ_ghk, θ_acceptreject, L)
    prim, res
end


function integrate_quardrature(prim::Primitives, ftn, upper_bound1, d=1, upper_bound2=nothing)
    @unpack KPU_1d, KPU_2d = prim

    integral = 0.0

    if d==1
        nodes = log.(KPU_1d[:, 1]) .+ upper_bound1
        integral = sum(KPU_1d[:, 2] .* ftn.(nodes) .* (1 ./ KPU_1d[:, 1]))

    elseif d==2
        nodes1 = log.(KPU_2d[:, 1]) .+ upper_bound1
        nodes2 = log.(KPU_2d[:, 2]) .+ upper_bound2
        integral = sum(KPU_2d[:, 3] .* ftn.(nodes1, nodes2) .* (1 ./ KPU_2d[:, 1]) .* (1 ./ KPU_2d[:, 2]))
    end

    return integral
end


function loglikelihood_quardrature(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64})

    α_0 = θ[1]
    α_1 = θ[2]
    α_2 = θ[3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]

    N = size(X, 1)
    σ_0 = 1 / (1 - ρ)^2
    L = zeros(N)

    @threads for i_index = 1:N

        x = XX[i_index, :]
        z = ZZ[i_index, :]
        t = TT[i_index]

        if t == 1.0
            L[i_index] = 1 - cdf(Normal(0, 1), ((-α_0 - x' * β - z[1] * γ) / σ_0))

        elseif t == 2.0
            m_2(ϵ0) = (1 - cdf(Normal(0, 1), (-α_1 .- x' * β .- z[2] * γ .- ρ * ϵ0))) * (pdf(Normal(0, 1), (ϵ0 / σ_0)) / σ_0)
            L[i_index] = integrate_quardrature(prim, m_2, -α_0 - x' * β - z[1] * γ, 1)

        elseif t == 3.0
            m_3(ϵ0, ϵ1) = (1 - cdf(Normal(0, 1), (-α_2 .- x' * β .- z[3] * γ .- ρ * ϵ1))) * pdf(Normal(0, 1), (ϵ1 - ρ * ϵ0)) * (pdf(Normal(0, 1), (ϵ0 / σ_0)) / σ_0)
            L[i_index] = integrate_quardrature(prim, m_3, -α_0 - x' * β - z[1] * γ, 2, -α_1 - x' * β - z[2] * γ)

        elseif t == 4.0
            m_4(ϵ0, ϵ1) = cdf(Normal(0, 1), (-α_2 .- x' * β .- z[3] * γ .- ρ * ϵ1)) * pdf(Normal(0, 1), (ϵ1 - ρ * ϵ0)) * (pdf(Normal(0, 1), (ϵ0 / σ_0)) / σ_0)
            L[i_index] = integrate_quardrature(prim, m_4, -α_0 - x' * β - z[1] * γ, 2, -α_1 - x' * β - z[2] * γ)
        end

    end

    logL = sum(log.(L))

    res.L = L

    return L, logL
end


function halton(base::Int64, n::Int64)

    m, d = 0, 1
    halton = zeros(n)

    for n_index = 1:n
        x = d - m
        if x == 1
            m = 1
            d *= base
        else
            y = d / base
            while x <= y
                y /= base
            end
            m = (base + 1) * y - x
        end
        halton[n_index] = m / d
    end

    halton
end


function loglikelihood_ghk(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64}, u_0, u_1)

    α_0 = θ[1]
    α_1 = θ[2]
    α_2 = θ[3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]

    n_draw = size(u_0, 1)

    σ_0 = 1 / (1 - ρ)^2
    N = size(X, 1)
    L = zeros(N)

    @threads for i_index = 1:N
        x = XX[i_index, :]
        z = ZZ[i_index, :]
        t = TT[i_index]

        truncation_0 = cdf(Normal(0, 1), ((-α_0 - x' * β - z[1] * γ) / σ_0))

        if t == 1.0

            L[i_index] = 1 - truncation_0

        else # if t = 2.0 or 3.0 or 4.0

            pr_0 = u_0 * truncation_0
            ε_0 = quantile.(Normal(0, σ_0), pr_0)

            truncation_1 = cdf(Normal(0, 1), -α_1 .- x' * β .- z[2] * γ .- ρ .* ε_0)

            if t == 2.0

                L[i_index] = sum((truncation_0 * ones(n_draw)) .* (1 .- truncation_1)) / n_draw

            else

                pr_1 = u_1 * truncation_1'
                η_1 = quantile.(Normal(0, 1), pr_1)
                ε_1 = ρ .* (ones(n_draw) * ε_0') .+ η_1

                truncation_2 = cdf(Normal(0, 1), -α_2 .- x' * β .- z[3] * γ .- ρ .* ε_1)

                if t == 3.0

                    L[i_index] = sum((truncation_0 * ones(n_draw, n_draw)) .* (ones(n_draw) * truncation_1') .* (1 .- truncation_2)) / (n_draw * n_draw)

                elseif t == 4.0

                    L[i_index] = sum((truncation_0 * ones(n_draw, n_draw)) .* (ones(n_draw) * truncation_1') .* (truncation_2)) / (n_draw * n_draw)

                end
            end
        end
        # if mod(i_index, 5000) == 0
        #     pct = i_index / N * 100
        #     println(pct)
        # end
    end

    logL = sum(log.(L))

    res.L = L

    return L, logL
end


function loglikelihood_acceptreject(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64}, u_0, u_1, u_2)

    α_0 = θ[1]
    α_1 = θ[2]
    α_2 = θ[3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]

    σ_0 = 1 / (1 - ρ)^2
    N = size(X, 1)
    L = zeros(N)

    ϵ_0 = quantile.(Normal(0, σ_0), u_0)
    η_1 = quantile.(Normal(0, 1), u_1)
    η_2 = quantile.(Normal(0, 1), u_2)
    ϵ_1 = ρ * ϵ_0 + η_1
    ϵ_2 = ρ * ϵ_1 + η_2

    for i_index = 1:N

        x = XX[i_index, :]
        z = ZZ[i_index, :]
        t = TT[i_index]

        if t == 1.0
            L[i_index] = mean( (ϵ_0 .> -α_0 - x' * β - z[1] * γ) )

        elseif t == 2.0
            L[i_index] = mean( (ϵ_0 .< -α_0 - x' * β - z[1] * γ) .* (ϵ_1 .> -α_1 - x' * β - z[2] * γ) )

        elseif t == 3.0
            L[i_index] = mean( (ϵ_0 .< -α_0 - x' * β - z[1] * γ) .* (ϵ_1 .< -α_1 - x' * β - z[2] * γ) .* (ϵ_2 .> -α_2 - x' * β - z[3] * γ) )

        elseif t == 4.0
            L[i_index] = mean( (ϵ_0 .< -α_0 - x' * β - z[1] * γ) .* (ϵ_1 .< -α_1 - x' * β - z[2] * γ) .* (ϵ_2 .< -α_2 - x' * β - z[3] * γ) )

        end

        # if mod(i_index, 5000) == 0
        #     pct = i_index / N * 100
        #     println(pct)
        # end
    end

    logL = sum(log.(L))

    res.L = L

    return L, logL
end


function integrate_quardrature2(prim::Primitives, ftn, lower_bound1, d=1, lower_bound2=nothing) # Quadrature integration from a to ∞.
    @unpack KPU_1d, KPU_2d = prim

    if d == 1
        nodes = -log.(1 .- KPU_1d[:, 1]) .+ lower_bound1
        integral = sum(KPU_1d[:, 2] .* ftn.(nodes) .* (1 ./ (1 .- KPU_1d[:, 1])))
    elseif d == 2
        nodes1 = -log.(1 .- KPU_2d[:, 1]) .+ lower_bound1
        nodes2 = -log.(1 .- KPU_2d[:, 2]) .+ lower_bound2
        integral = sum(KPU_2d[:, 3] .* ftn.(nodes1, nodes2) .* (1 ./ (1 .- KPU_2d[:, 1])) .* (1 ./ (1 .- KPU_2d[:, 2])))
    end
    return integral
end


function loglikelihood_quardrature_alt(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64}) # Directly follows the formula provided in PS2.

    α_0 = θ[1]
    α_1 = θ[2]
    α_2 = θ[3]
    β = θ[4:18]
    γ = θ[19]
    ρ = θ[20]

    N = size(X, 1)
    σ_0 = 1 / (1 - ρ)^2
    L = zeros(N)

    @threads for i_index = 1:N
        x = XX[i_index, :]
        z = ZZ[i_index, :]
        t = TT[i_index]

        if t == 1.0
            L[i_index] = cdf(Normal(0, 1), ((-α_0 - x' * β - z[1] * γ) / σ_0))

        elseif t == 2.0
            m_2(ϵ0) = cdf(Normal(0, 1), (-α_1 .- x' * β .- z[2] * γ .- ρ * ϵ0)) * pdf(Normal(0, 1), (ϵ0 / σ_0)) * (1 / σ_0)
            L[i_index] = integrate_quardrature(prim, m_2, α_0 + x' * β + z[1] * γ, 1)

        elseif t == 3.0
            m_3(ϵ0, ϵ1) = cdf(Normal(0, 1), (-α_2 .- x' * β .- z[3] * γ .- ρ * ϵ1)) * pdf(Normal(0, 1), (ϵ1 - ρ * ϵ0)) * pdf(Normal(0, 1), (ϵ0 / σ_0)) * (1 / σ_0)
            L[i_index] = integrate_quardrature(prim, m_3, α_0 + x' * β + z[1] * γ, 2, α_1 + x' * β + z[2] * γ)

        elseif t == 4.0
            m_4(ϵ0, ϵ1) = cdf(Normal(0, 1), (α_2 .+ x' * β .+ z[3] * γ .- ρ * ϵ1)) * pdf(Normal(0, 1), (ϵ1 - ρ * ϵ0)) * pdf(Normal(0, 1), (ϵ0 / σ_0)) * (1 / σ_0)
            L[i_index] = integrate_quardrature(prim, m_4, α_0 + x' * β + z[1] * γ, 2, α_1 + x' * β + z[2] * γ)
        end

    end

    logL = sum(log.(L))

    return L, logL
end