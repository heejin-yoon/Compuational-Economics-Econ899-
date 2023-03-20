##

@with_kw struct Primitives
    T::Int64 = 200
    H::Int64 = 10
    ρ₀::Float64 = 0.5
    σ₀::Float64 = 1.0
end

##

mutable struct Estimates
    ϵ::Array{Float64}
    x_true::Array{Float64}
    m_true::Array{Float64}
    x::Array{Float64}
    m::Array{Float64}
    b_hat1::Array{Float64}
    b_hat2::Array{Float64}
end

##

function Initialize()
    prim = Primitives()
    ϵ = zeros(prim.T, prim.H)
    x_true = zeros(prim.T)
    m_true = zeros(3, 1)
    x = zeros(prim.T, prim.H)
    m = zeros(3, 1)
    b_hat1 = zeros(2, 1)
    b_hat2 = zeros(2, 1)
    est = Estimates(ϵ, x_true, m_true, x, m, b_hat1, b_hat2)
    prim, est                                                                    # return deliverables
end

##

function Truedata(prim::Primitives)
    @unpack T, H, ρ₀, σ₀ = prim

    Random.seed!(12341234)
    dist = Normal(0, σ₀)
    x = zeros(T)

    x[1] = rand(dist)
    for T_index = 2:T
        x[T_index] = ρ₀*x[T_index-1] + rand(dist)
    end

    m = zeros(3)
    m[1] = mean(x)
    m[2] = sum((x .- mean(x)).^2)/T
    m[3] = sum((x[2:T] .- mean(x)) .* (x[1:T-1] .- mean(x)))/T

    est.x_true = x
    est.m_true = m

    x, m
end

##

function Simulatedata(prim::Primitives, b::Array{Float64})
    @unpack T, H = prim

    ρ = b[1]
    σ = b[2]
    Random.seed!(12341234)
    dist = Normal(0, σ)
    ϵ = rand(dist, T, H)
    x = zeros(T, H)

    for H_index = 1:H
        x[1, H_index] = ϵ[1, H_index]
        for T_index = 2:T
            x[T_index, H_index] = ρ*x[T_index-1, H_index] + ϵ[T_index, H_index]
        end
    end

    x̄ = zeros(1, H)
    for H_index = 1:H
        x̄[1, H_index] = mean(x[:, H_index])
    end

    m = zeros(3)
    m[1] = mean(x)
    m[2] = mean((x .- x̄).^2)
    m[3] = mean((x .- x̄)[2:T, :] .* (x .- x̄)[1:T-1, :])

    ϵ, x, m
end

##

function est_first(prim::Primitives, est::Estimates, W::Matrix{Float64})
    m_hat(ρ, σ) = Simulatedata(prim, ρ, σ)[3]
    J(ρ, σ) = (est.m - m_hat(ρ, σ))' * W * (est.m - m_hat(ρ, σ))
    lower = [0.35, 0.8]
    upper = [0.65, 1.2]
    b_initial = (lower + upper)./2
    b_hat = optimize(b -> J(b[1], b[2]), [prim.ρ₀, prim.σ₀]).minimizer

    b_hat
end

##

function ComputeSE(prim::Primitives, ρ_hat::Float64, σ_hat::Float64, W::Array{Float64}, i::Int64, j::Int64)
    @unpack T = prim

    δ = 1e-10
    x1, m1 = GetMoments(prim, ρ_hat, σ_hat)
    x2, m2 = GetMoments(prim, ρ_hat - δ, σ_hat)
    x3, m3 = GetMoments(prim, ρ_hat, σ_hat - δ)
    g1 = m1[i:j]
    g2 = m2[i:j]
    g3 = m3[i:j]
    ▽ρ = (g2 .- g1)./δ # bigtriangledown
    ▽σ = (g3 .- g1)./δ
    ▽b = [▽ρ ▽σ]
    Σ_hat = sqrt.(diag(1/T*inv(▽b' * W * ▽b)))
    Σ_hat
end

##

function NeweyWest(prim::Primitives, est::Estimates)
    @unpack H, T = prim

    max_lag = 4
    x, m_TH = Simulatedata(prim, est.b_hat1[1], est.b_hat1[2])[2:3]
    Γ = zeros(3, 3, max_lag+1)
    S_y = zeros(3, 3)
    S = zeros(3, 3)
    for j = 0:max_lag
        for H_index = 1:H
            for t_index = 1+j:T
                y_th = x[t_index, H_index]
                if t_index == 1
                    y_th_lag = 0.0
                else
                    y_th_lag = x[t_index-1, H_index]
                end
                m = zeros(3)
                m[1] = y_th
                m[2] = (y_th - mean(x[:, H_index]))^2
                m[3] = (y_th  - mean(x[:, H_index])) * (y_th_lag  - mean(x[:, H_index]))
                Γ[:, :, j+1] += (est.m - m_TH) * (est.m - m_TH)' ./ (T * H)
            end
        end
    end

    S_y = Γ[:, :, 1]

    for j = 1:max_lag
        S_y += (1 - (j / (max_lag + 1))) * (Γ[:, :, j+1] + Γ[:, :, j+1]')
    end

    S = (1 + 1/H).*S_y

    S
end

##

function SMM(prim::Primitives, est::Estimates, i::Int64, j::Int64)
    @unpack T, H = prim

    W = Matrix{Float64}(I, j-i+1, j-i+1)
    ρ_hat1, σ_hat1 = MinObjFunc(prim, est, W, i, j)
    est.ρ_hat1 = ρ_hat1
    est.σ_hat1 = σ_hat1

    Σ_hat1 = ComputeSE(prim, ρ_hat1, σ_hat1, W, i, j)
    S = NeweyWest(prim, est, i, j)
    W_star = inv(S)
    ρ_hat2, σ_hat2 = MinObjFunc(prim, est, W_star, i, j)

    Σ_hat2 = ComputeSE(prim, ρ_hat2, σ_hat2, W_star, i, j)

    J = ObjFunc(prim, ρ_hat2, σ_hat2, W_star, i, j)
    J_stat = ((T*H)/(1+H))*J
    J_stat_p = cdf(Chisq(1), J_stat)

    println("")
    println("************************************************")
    println("Step1: Estimate of b with I: ", [ρ_hat1, σ_hat1])
    println("Step2: Estimate of b with W*: ", [ρ_hat2, σ_hat2])
    println("Standard errors: ", Σ_hat2)
    println("J-test statistic: ", J_stat, " (p-value: ", J_stat_p, ")")
    println("************************************************")

    ρ_hat1, σ_hat1, Σ_hat1, ρ_hat2, σ_hat2, Σ_hat2, J_stat, J_stat_p
end
