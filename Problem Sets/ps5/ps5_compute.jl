## Open Packages and Files

using Interpolations, Plots, Parameters
include("ps5_model_workinprogress.jl");                                                         # import the functions that solve our growth model
prim, res = Initialize()
idio_state, agg_state = draw_shocks(prim, res)
V_iterate(prim, res)

##

KK_path = zeros(prim.T) # aggregate capital path
k_path = zeros(prim.N, prim.T)
KK_path[1] = 11.55
for n = 1:prim.N
    k_path[n, 1] = KK_path[1]
end
pol_func_interp = interpolate(res.pol_func, BSpline(Linear()))
# k_choices_next = ones(prim.N)
for t = 1:15
    KK_index = get_index(KK_path[t], prim.K_grid)
    z_index = agg_state[t]
    idio_state_t = idio_state[:, t]
    k_path_t = k_path[:, t]
    for n = 1:prim.N
        ε_index = idio_state_t[n]
        k_index = get_index(k_path_t[n], prim.k_grid)
        k_path[t+1, n] = pol_func_interp[k_index, ε_index, KK_index, z_index]
        KK_path[t+1] += k_path[t+1, n]
        println("N: ", n, " and T: ", t, " is done.")
    end
    KK_path[t+1] = KK_path[t+1]/prim.N
end

@elapsed KK_path, k_path = KKpath(prim, res)

#%% Test

Y = zeros(Prim.T - 1000, 1)
X = zeros(Prim.T - 1000, 2)
Y[:, 1] = KK_path[1001:Prim.T]

for t = 1:(prim.T-1000)
    if agg_state[t-1+P.burn]==1 # good state
        X[t, 1] = KK_path[tt-1+P.burn]
    else
        X[t, 2] = KK_path[tt-1+P.burn]
    end
end

Y = zeros(20, 1)
X = zeros(20, 2)
Y[:, 1] = KK_path[2:21]

for t = 1:20
    if agg_state[t]==1
        X[t, 1] = KK_path[t]
    elseif agg_state[t]==2
        X[t, 2] = KK_path[t]
    end
end
