import pandas as pd
import numpy as np
import time
import xpress as xp # FICO Xprerss Solver
xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

from atcs import ATCS

def run_minlp_deterministic(data:ATCS, N_sample, sample_type, seed = 1,save = True):
    # MINLP Model
    # Data
    robot_id = data.r_sub_df.index.to_numpy()
    robot_loc = data.l_sub_df.to_numpy()
    robot_range = data.r_sub_df.to_numpy().flatten()
    x_min,x_max = np.min(robot_loc, axis=0)[0], np.max(robot_loc, axis=0)[0]
    y_min,y_max = np.min(robot_loc, axis=0)[1], np.max(robot_loc, axis=0)[1]
    dist_matrix = data.get_distance_matrix(sample_subset = True)

    # Creates an empty Model
    model = xp.problem(name='MINLP_Deterministic')

    # Given Inputs and Parameters
    R = data.r_sub_df['range'].to_dict()
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

    # Parameters
    U_Station = len(robot_id)
    data.set_output_folder(solver_type, model_type, name = f'{N_sample}_{sample_type}samp_seed{seed}')

    # U_Station = N_sample
    # Sets and Notation
    V = data.r_sub_df.index.to_numpy() # Robot Vehicle sets
    S = np.arange(U_Station)


    # Output Variables
    x = model.addVariables(S, name = 'x', vartype = xp.continuous, lb = x_min, ub = x_max) # Longitude of station s
    y = model.addVariables(S, name = 'y', vartype = xp.continuous, lb = y_min, ub = y_max) # Latitude of station s

    eta = model.addVariables(S, name = 'eta', vartype = xp.integer) # Number of charger installed at station s
    mu = model.addVariables(S, name = 'mu', vartype = xp.binary) # Whether station s is built
    beta = model.addVariables(V, S, name = 'beta', vartype = xp.binary) # Whether robot v is covered by station s
    alpha = model.addVariables(V, S, name = 'alpha', vartype = xp.binary) # Whether robot v is brought to station s by taking penalty
    d = model.addVariables(V, S, name = 'd', vartype = xp.continuous, lb = 0) # Distance from v to s


    # Constraints
    # Robot v can be allocated to charging station s iff it is built
    model.addConstraint(beta[v,s] <= mu[s] for v in V for s in S) 

    model.addConstraint(xp.Sum(mu[s] for s in S) >= L_Station)
    # Each Robot v must be allocated to only one station
    model.addConstraint(xp.Sum(beta[v,s] for s in S) == 1 for v in V)

    # Number of chargers per station
    model.addConstraint(eta[s] >= xp.Sum(beta[v,s] for v in V)/U_Robot for s in S)
    model.addConstraint(eta[s] <= U_Charger for s in S)
    # Robot v can be brought to station s by human iff the robot is allocated to that station
    model.addConstraint(alpha[v,s] <= beta[v,s] for v in V for s in S)

    # Distance from robot v to station s
    # If Robot v is allocated to station s,
    #       If its range can reach s, alpha = 0
    #       Else; distance is 0 and it is brought to s by taking penalty (alpha = 1)
    # Else; d[v,s] = 0
    model.addConstraint(d[v,s] <= R[v] * (1 - alpha[v,s]) for v in V for s in S)
    model.addConstraint(d[v,s] >= xp.sqrt((Long_V[v] - x[s])**2 + (Lat_V[v] - y[s])**2)
                                    - Dist_Max * (alpha[v,s] + (1 - beta[v,s])) 
                                            for v in V for s in S)

    model.addConstraint(d[v,s] <= Dist_Max * beta[v,s] for v in V for s in S)

    # Objective Function
    Build_station_cost = xp.Sum(C_b * mu[s] for s in S)
    Build_charger_cost = xp.Sum(C_m * eta[s] for s in S)
    Penalty_cost = xp.Sum(C_h * alpha[v,s] for v in V for s in S)
    Charging_cost = xp.Sum(C_c * (beta[v,s]*U_Range - (beta[v,s]*R[v] - d[v,s])) for v in V for s in S)

    obj = Build_station_cost + Build_charger_cost + Charging_cost + Penalty_cost

    model.setObjective(obj, sense=xp.minimize)
    # model.setControl('miprelstop', 1e-3)
    model.setControl('maxtime', 300)
    tic = time.time()
    # Solve the problem
    model.solve()
    toc=  time.time()
    run_time = toc - tic

    # Processed the Output
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

    robot_df = pd.DataFrame([beta_dict, alpha_dict, d_dict],['beta','alpha','d']).T.reset_index()
    robot_df.rename(columns={'index':'(v,s)'}, inplace=True)
    robot_df['v'], robot_df['s'] = robot_df['(v,s)'].apply(lambda x: x[0]), robot_df['(v,s)'].apply(lambda x: x[1])
    robot_df.drop(['(v,s)'], axis = 1, inplace= True)
    robot_df = robot_df[['v','s','beta','alpha','d']]

    if save:
        station_df.to_csv(f'{data.folder_name}/station_df.csv')
        robot_df.to_csv(f'{data.folder_name}/robot_df.csv')

    return summary_df, station_df, robot_df, mu_dict, eta_dict, x_dict, y_dict, beta_dict, alpha_dict, d_dict


if __name__ == "__main__":
    for N_sample in [25, 50, 75, 100, 125, 150]:
        print(f'N_sample: {N_sample}')
        # N_sample = 50
        seed = 1
        solver_type = 'minlp'
        model_type = 'deterministic'
        sample_type = f'head'
        data = ATCS(seed = 1)
        data.choose_subset_point(N_sample, randomized = False) # Choose subset data

        (summary_df, station_df, robot_df, 
        mu_dict, eta_dict, x_dict, y_dict, 
        beta_dict, alpha_dict, d_dict) = run_minlp_deterministic(data, N_sample, sample_type, seed = 1,save = True)

