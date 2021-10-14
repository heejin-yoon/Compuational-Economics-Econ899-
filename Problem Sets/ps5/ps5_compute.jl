## Open Packages and Files

using Interpolations, Plots, Parameters
include("ps5_model_hj2.jl");                                                         # import the functions that solve our growth model
prim, res = Initialize()
idio_state, agg_state = draw_shocks(prim)
V_iterate(prim, res)

##


function Kpath()
K_path = zeros(prim.T) # aggregate capital path
K_ss_today = 11.55 # initial value
k_choices = K_ss_today .* ones(prim.N) # initially all k choices of individuals are the same as K.
K_index = get_index(K_ss_today, prim.K_grid) # get index of K
k_index = ones(prim.N) * get_index(K_ss_today, prim.k_grid) # get index of k
pol_func_interp = interpolate(res.pol_func, BSpline(Linear()))
    for t = 1:5
    # for t = 1:prim.T
        K_path[t] = K_ss_today
        for n = 1:prim.N
            ε_index = idio_state[n, t]
            z_index = agg_state[t]
            k_choices[n] = pol_func_interp[k_index[n], ε_index, K_index, z_index]
            k_index[n] = get_index(k_choices[n], prim.k_grid)
            println("N: ", n, " and T: ", t, " is done.")
        end
        K_ss_today = mean(k_choices)
        K_index = get_index(K_ss_today, prim.K_grid)
    end
end

@elapsed Kpath()
