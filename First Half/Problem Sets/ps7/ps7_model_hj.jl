# ------------------------------------------------------------------------------
# Author: Heejin
# Hopenhayn and Rogerson (1993, JPE)
# October 21, 2021
# ps6_compute.jl
# ------------------------------------------------------------------------------



# keyword-enabled structure to hold model Primitives

@with_kw struct Primitives
    T::Int64 = 200
    H::Int64 = 10
    ρ₀::Float64 = 0.5                                                            # discount rate
    σ₀::Float64 = 1.0    #                                                        # coefficient of relative risk aversion
end

# structure that holds model results

mutable struct Estimates
    ϵ::Array{Float64}
    x_true::Array{Float64}
    m_true::Array{Float64}
    x::Array{Float64}
    m::Array{Float64}
    ρ_hat1::Float64
    σ_hat1::Float64
    ρ_hat2::Float64
    σ_hat2::Float64
end

function Initialize()
    prim = Primitives()
    ϵ = zeros(prim.T, prim.H)
    x_true=zeros(prim.T) #
    m_true=zeros(3, 1) #
    x = zeros(prim.T, prim.H)
    m = zeros(3, 1)
    ρ_hat1 = 0.0
    σ_hat1 = 0.0
    ρ_hat2 = 0.0
    σ_hat2 = 0.0
    est = Estimates(ϵ, x_true, m_true, x, m, ρ_hat1, σ_hat1, ρ_hat2, σ_hat2)
    prim, est                                                                    # return deliverables
end

function Truedata(prim::Primitives)
    @unpack T, H, ρ₀, σ₀ = prim

    # Simulate H AR(1) process of length T
    Random.seed!(12032020)
    dist = Normal(0, σ₀)
    x = zeros(T)

    x[1] = rand(dist)
    for T_index = 2:T
        x[T_index] = ρ₀*x[T_index-1] + rand(dist)
    end

    m = zeros(3)
    m[1] = mean(x)
    m[2] = var(x)
    m[3] = autocor(x, [1]; demean = true)[1]

    est.x_true = x
    est.m_true = m

    x, m
end

function GetMoments(prim::Primitives, ρ::Float64, σ::Float64)
    @unpack T, H = prim

    # Simulate H AR(1) process of length T
    Random.seed!(12032020)
    dist = Normal(0, σ)
    ϵ = rand(dist, T, H)
    x = zeros(T, H)

    est.ϵ = ϵ
    est.x = x

    for H_index = 1:H
        x[1, H_index] = ϵ[1, H_index]
        for T_index = 2:T
            x[T_index, H_index] = ρ*x[T_index-1, H_index] + ϵ[T_index, H_index]
        end
    end

    # Compute appropriate moments for each of the T simulations.
    x_mean = zeros(1, H)
    x_var = zeros(1, H)
    x_autocorr = zeros(1, H)
    x_temp = zeros(T, H)

    for H_index = 1:H
        x_mean[H_index] = sum(x[:, H_index])/T
        x_temp[:, H_index] = x[:, H_index] .- x_mean[1, H_index]
        x_var[H_index] = sum(x_temp[:, H_index] .* x_temp[:, H_index])/T
        x_autocorr[H_index] = sum(x_temp[1:T-1, H_index] .* x_temp[2:T, H_index])/T
    end

    # Average across H individual moments
    m = zeros(3)
    m[1] = mean(x_mean)
    m[2] = mean(x_var)
    m[3] = mean(x_autocorr)

    est.x = x
    est.m = m

    x, m
end

function ObjFunc(prim::Primitives, ρ::Float64, σ::Float64, W::Array{Float64}, i::Int64, j::Int64)

    x₀, m₀ = Truedata(prim)
    x_hat, m_hat = GetMoments(prim, ρ, σ)

    g₀ = m₀[i:j]
    g_hat = m_hat[i:j]
    J = (g₀ - g_hat)' * W * (g₀ - g_hat)

    J
end

function MinObjFunc(prim::Primitives, est::Estimates, W::Array{Float64}, i::Int64, j::Int64)
    @unpack ρ₀, σ₀ = prim

    # W = Matrix{Float64}(I)
    ρ_hat, σ_hat = optimize(b->ObjFunc(prim, b[1], b[2], W, i, j), [prim.ρ₀, prim.σ₀]).minimizer

    ρ_hat, σ_hat
end

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

function NeweyWest(prim::Primitives, est::Estimates, i::Int64, j::Int64)
    @unpack H = prim

    lag_max = 4
    Sy = GammaFunc(prim, est, 0, i, j)

    # loop over lags
    for n = 1:lag_max
        gamma_n = GammaFunc(prim, est, n, i, j)
        Sy += (gamma_n + gamma_n').*(1-(n/(lag_max + 1)))
    end
    S = (1 + 1/H).*Sy

    return S
end

function GammaFunc(prim::Primitives, est::Estimates, lag::Int64, i::Int64, j::Int64)
    @unpack H, T = prim
    @unpack x, m = est

    gamma_tot = zeros(3, 3)

    for T_index = (1+lag):T
        for H_index = 1:H
            # No Lagged
            avg_obs = x[T_index, H_index]
            if T_index > 1
                avg_obs_tm1 = x[T_index-1, H_index]
            else
                avg_obs_tm1 = 0
            end
            avg_h = mean(x[:, H_index])
            var_obs = (avg_obs - avg_h)^2
            auto_cov_obs = (avg_obs - avg_h)*(avg_obs_tm1 - avg_h)

            mom_obs_diff = [avg_obs, var_obs, auto_cov_obs] - m
            mom_obs_diff = mom_obs_diff

            # Lagged
            avg_lag = x[T_index-lag, H_index]
            if T_index - lag > 1
                avg_lag_tm1 = x[T_index-lag-1, H_index]
            else
                avg_lag_tm1 = 0
            end
            avg_h = mean(x[:, H_index])
            var_lag = (avg_lag - avg_h)^2
            auto_cov_lag = (avg_lag - avg_h)*(avg_lag_tm1 - avg_h)


            mom_lag_diff = [avg_lag, var_lag, auto_cov_lag] - m
            mom_lag_diff = mom_lag_diff

            gamma_tot += mom_obs_diff*mom_lag_diff'
        end
    end

    gamma = (1/(T*H)).*gamma_tot

    gamma[i:j, i:j]
end


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
