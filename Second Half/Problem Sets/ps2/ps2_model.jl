@with_kw struct Primitives
    data = DataFrame(load(rt * "/Problem Sets/ps2/Mortgage_performance_data.dta"))
    KPU_1d::Array{Float64} = Array(DataFrame(CSV.File(rt * "/Problem Sets/ps2/KPU_d1_l20.csv")))
    KPU_2d::Array{Float64} = Array(DataFrame(CSV.File(rt * "/Problem Sets/ps2/KPU_d2_l20.csv")))
end

mutable struct Results
    θ::Array{Float64}
end

function Initialize()
    prim = Primitives()
    α_0 = 0.0
    α_1 = -1.0
    α_2 = -1.0
    β = Array(fill(0.0, 15))
    γ = 0.3
    ρ = 0.5
    θ = zeros(20)
    θ = [α_0; α_1; α_2; β; γ; ρ]
    res = Results(θ)
    prim, res
end


function integrate_quardrature(prim::Primitives, ftn, upper_bound1, d=1, upper_bound2=nothing)
    @unpack KPU_1d, KPU_2d = prim

    if d == 1
        nodes = log.(KPU_1d[:, 1]) .+ upper_bound1
        integral = sum(KPU_1d[:, 2] .* ftn.(nodes) .* (1 ./ KPU_1d[:, 1]))
    elseif d == 2
        nodes1 = log.(KPU_2d[:, 1]) .+ upper_bound1
        nodes2 = log.(KPU_2d[:, 2]) .+ upper_bound2
        integral = sum(KPU_2d[:, 3] .* ftn.(nodes1, nodes2) .* (1 ./ KPU_2d[:, 1]) .* (1 ./ KPU_2d[:, 2]))
    end
    return integral
end




function loglikelihood_quardrature(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64})
    # println(θ)

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


function integrate_quardrature_alt(prim::Primitives, ftn, lower_bound1, d=1, lower_bound2=nothing)
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



function loglikelihood_quardrature_alt(prim::Primitives, θ::Array{Float64}, XX::Array{Float64}, ZZ::Array{Float64}, TT::Array{Float64})
    # println(θ)

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
            L[i_index] = integrate_quardrature_alt(prim, m_4, -α_0 - x' * β - z[1] * γ, 2, -α_1 - x' * β - z[2] * γ)
        end
    end

    logL = sum(log.(L))

    return L, logL
end

