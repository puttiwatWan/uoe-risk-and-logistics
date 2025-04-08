import pandas as pd
import numpy as np
import time
import xpress as xp # FICO Xprerss Solver
xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

from atcs import ATCS

def run_minlp_stochastic(data:ATCS, N_sample, sample_type, seed = 1,save = True):
    # Stochastic MINLP Model
    # Data
    robot_id = data.r_s_sub_df.index.to_numpy()
    robot_loc = data.l_sub_df.to_numpy()
    # robot_range = data.R_Kub_df.to_numpy().flatten()
    x_min,x_max = np.min(robot_loc, axis=0)[0], np.max(robot_loc, axis=0)[0]
    y_min,y_max = np.min(robot_loc, axis=0)[1], np.max(robot_loc, axis=0)[1]
    # dist_matrix = data.get_distance_matrix(sample_subset = True)

    # Creates an empty Model
    model = xp.problem(name='MINLP_Stochastic')
    # Given Inputs and Parameters
    R_K = {(i, col): data.r_s_sub_df.at[i, col] for i in data.r_s_sub_df.index for col in data.r_s_sub_df.columns}
    L_Range = data.r_min # km
    U_Range = data.r_max # km
    U_Robot = data.q # Max robots per charger
    U_Charger = data.m # Max chargers per station
    L_Station = int(np.ceil(N_sample/(U_Charger * U_Robot))) # Maximum Robots per opened station
    Dist_Max = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)
    C_b = data.c_b # Investment cost per station
    C_h = data.c_h # Cost of moving a robot
    C_m = data.c_m # Maintenance cost per charger
    C_c = data.c_c # Charging cost per km

    Long_V, Lat_V = data.l_sub_df['longitude'].to_dict(), data.l_sub_df['latitude'].to_dict()
    A_dict = {(i, col): data.cc_df.at[i, col] for i in data.cc_df.index for col in data.cc_df.columns}

    # Parameters
    # U_Station = L_Station + 1 # For now
    U_Station = len(robot_id)
    data.set_output_folder(solver_type, model_type, name = f'{N_sample}_{sample_type}samp_seed{seed}')

    # Sets and Notation
    V = data.r_s_sub_df.index.to_numpy() # Robot Vehicle sets
    S = np.arange(U_Station)
    K = data.r_s_sub_df.columns.to_list()


    # Output Variables
    x = model.addVariables(S, name = 'x', vartype = xp.continuous, lb = x_min, ub = x_max) # Longitude of station s
    y = model.addVariables(S, name = 'y', vartype = xp.continuous, lb = y_min, ub = y_max) # Latitude of station s

    eta = model.addVariables(S, name = 'eta', vartype = xp.integer) # Number of charger installed at station s
    mu = model.addVariables(S, name = 'mu', vartype = xp.binary) # Whether station s is built
    beta = model.addVariables(V, S, K, name = 'beta', vartype = xp.binary) # Whether robot v is covered by station s
    alpha = model.addVariables(V, S, K, name = 'alpha', vartype = xp.binary) # Whether robot v is brought to station s by taking penalty
    d = model.addVariables(V, S, K, name = 'd', vartype = xp.continuous, lb = 0) # Distance from v to s


    # Robot v can be allocated to charging station s iff it is built
    model.addConstraint(beta[v,s,k] <= A_dict[v,k] * mu[s] for v in V for s in S for k in K) 

    model.addConstraint(xp.Sum(mu[s] for s in S) >= L_Station)
    # Each Robot v must be allocated to only one station
    model.addConstraint(xp.Sum(beta[v,s,k] for s in S) == 1 * A_dict[v,k] for v in V for k in K)

    # Number of chargers per station
    model.addConstraint(eta[s] >= xp.Sum(A_dict[v,k]*beta[v,s,k] for v in V)/U_Robot for s in S for k in K)
    model.addConstraint(eta[s] <= U_Charger for s in S)

    # Robot v can be brought to station s by human iff the robot is allocated to that station
    model.addConstraint(alpha[v,s,k] <= beta[v,s,k] * A_dict[v,k] for v in V for s in S for k in K)

    model.addConstraint(d[v,s,k] <= R_K[v,k] * (1 - alpha[v,s,k]) * A_dict[v,k] for v in V for s in S for k in K)
    model.addConstraint(d[v,s,k] >= xp.sqrt((Long_V[v] - x[s])**2 + (Lat_V[v] - y[s])**2)
                                    - Dist_Max * (alpha[v,s,k] + (1 - beta[v,s,k])) 
                                            for v in V for s in S for k in K)
    model.addConstraint(d[v,s,k] <= Dist_Max * beta[v,s,k] * A_dict[v,k] for v in V for s in S for k in K)

    # Objective Function
    Build_station_cost = xp.Sum(C_b * mu[s] for s in S)
    Build_charger_cost = xp.Sum(C_m * eta[s] for s in S)

    Penalty_cost = (1/len(K))*xp.Sum(C_h * alpha[v,s,k] * A_dict[v,k] for v in V for s in S for k in K)
    Charging_cost = (1/len(K))*xp.Sum(C_c * A_dict[v,k]*(beta[v,s,k]*U_Range - (beta[v,s,k]*R_K[v,k] - d[v,s,k])) for v in V for s in S for k in K)

    obj = Build_station_cost + Build_charger_cost + Charging_cost + Penalty_cost

    model.setObjective(obj, sense=xp.minimize)
    model.setControl('maxtime', 3600)
    
    # Solve the problem
    tic = time.time()
    model.solve()
    toc = time.time()
    run_time = tic - toc
    
    # Preprocess the Output
    obj_value = model.getObjVal()
    mu_dict = model.getSolution(mu)
    eta_dict = model.getSolution(eta)
    x_dict = model.getSolution(x)
    y_dict = model.getSolution(y)
    beta_dict = model.getSolution(beta)
    alpha_dict = model.getSolution(alpha)
    d_dict = model.getSolution(d)

    summary_df = pd.DataFrame({'Objective':[obj_value], 'RunTime':[run_time]})

    station_df = pd.DataFrame([mu_dict, eta_dict, x_dict, y_dict], ['mu','eta','longitude','latitude']).T
    station_df['Building_Station_Cost'] = station_df['mu'] * C_b
    station_df['Building_Charger_Cost'] = station_df['eta'] * C_m

    r_k_df = pd.DataFrame.from_dict(R_K, orient='index')
    r_k_df.rename(columns={0:'range_scenario'}, inplace=True)

    A_df = pd.DataFrame.from_dict(A_dict, orient='index')
    A_df.rename(columns={0:'A_v_k'}, inplace = True)

    robot_df = pd.DataFrame([beta_dict, alpha_dict, d_dict],['beta','alpha','d']).T.reset_index()
    robot_df.rename(columns={'index':'(v,s,k)'}, inplace=True)
    robot_df['v'], robot_df['s'], robot_df['k'] = robot_df['(v,s,k)'].apply(lambda x: x[0]), robot_df['(v,s,k)'].apply(lambda x: x[1]), robot_df['(v,s,k)'].apply(lambda x: x[2])
    robot_df['(v,k)'] = robot_df['(v,s,k)'].apply(lambda x: (x[0],x[2]))
    robot_df.drop(['(v,s,k)'], axis = 1, inplace= True)
    robot_df = robot_df.set_index('(v,k)').merge(r_k_df, left_index = True, right_index = True, how = 'left')
    robot_df = robot_df.merge(A_df, left_index = True, right_index = True, how = 'left')
    robot_df.reset_index(drop = True, inplace=True)
    
    if save:
        station_df.to_csv(f'{data.folder_name}/station_df.csv')
        robot_df.to_csv(f'{data.folder_name}/robot_df.csv')

    return summary_df, station_df, robot_df, mu_dict, eta_dict, x_dict, y_dict, beta_dict, alpha_dict, d_dict


if __name__ == "__main__":
    N_sample = 4
    seed = 1
    solver_type = 'minlp'
    model_type = 'stochastic'
    sample_type = f'head_{N_sample}'
    data = ATCS(seed = 1)
    data.choose_subset_point(N_sample, randomized = False) # Choose subset data

    (summary_df, station_df, robot_df, 
    mu_dict, eta_dict, x_dict, y_dict, 
    beta_dict, alpha_dict, d_dict) = run_minlp_stochastic(data, N_sample, sample_type, seed = 1,save = True)
