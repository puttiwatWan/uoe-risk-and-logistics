import preprocessing as pp
import pandas as pd
import numpy as np
import time 
import xpress as xp # FICO Xprerss Solver
xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

def sto_model(  agg_dem_cus_period_clus_scene_df:pd.DataFrame,
                candidate_df:pd.DataFrame,
                supplier_df:pd.DataFrame,
                cluster_center_df:pd.DataFrame,
                cost_w_to_s:pd.DataFrame,
                cost_w_to_cluster:pd.DataFrame,
                time_limit_s:int = 3600):
                
    model = xp.problem(name= 'MEWLP_Stochastics')
    # Sets and Notation
    S = list(supplier_df.index) # Supplier Index
    S_P0 = list(supplier_df[supplier_df['SupplierProductGroup'] == 0]['SupplierProductGroup'].index)
    S_P1 = list(supplier_df[supplier_df['SupplierProductGroup'] == 1]['SupplierProductGroup'].index)
    S_P2 = list(supplier_df[supplier_df['SupplierProductGroup'] == 2]['SupplierProductGroup'].index)
    S_P3 = list(supplier_df[supplier_df['SupplierProductGroup'] == 3]['SupplierProductGroup'].index)
    S_P_dict = {0:S_P0, 1:S_P1, 2:S_P2, 3:S_P3}

    W = list(candidate_df.index) # Warehouse Index
    C = list(cluster_center_df.index) # Cluster Index
    P = list(agg_dem_cus_period_clus_scene_df.reset_index()['ProductIndex'].unique())
    T = list(agg_dem_cus_period_clus_scene_df.reset_index()['PeriodIndex'].unique())
    Phi = list(agg_dem_cus_period_clus_scene_df.columns)

    # Output Variables
    x = np.array([xp.var(f'x_{w}_{c}_{t}_{xi}', vartype = xp.binary) for w in W for c in C for t in T for xi in Phi], dtype = xp.npvar).reshape(len(W), len(C), len(T), len(Phi))
    y = np.array([xp.var(f'y_{w}_{t}', vartype = xp.binary) for w in W for t in T], dtype = xp.npvar).reshape(len(W), len(T))
    o = np.array([xp.var(f'o_{w}', vartype = xp.binary) for w in W], dtype = xp.npvar).reshape(len(W))
    v = np.array([xp.var(f'v_{w}_{c}_{p}_{t}_{xi}', vartype = xp.continuous, lb = 0) for w in W for c in C for p in P for t in T for xi in Phi], dtype = xp.npvar).reshape(len(W), len(C), len(P), len(T), len(Phi))
    z = np.array([xp.var(f'z_{w}_{s}_{t}', vartype = xp.continuous, lb = 0) for w in W for s in S for t in T], dtype = xp.npvar).reshape(len(W), len(S), len(T))

    model.addVariable(x, y, o, v, z)


    # Constraints 
    for w in W:
        for t in T:
            model.addConstraint(xp.constraint(o[w] >= y[w,t]))
            if t != 0:
                model.addConstraint(xp.constraint(y[w, t] >= y[w,t-1]))    

            for c in C:
                for xi in Phi:
                    model.addConstraint(xp.constraint(x[w,c,t,xi] <= y[w,t]))  

    for xi in Phi:
        for t in T:
            for w in W:
                Capacity_W = candidate_df.loc[w,'Capacity']
                model.addConstraint(xp.constraint(xp.Sum(v[w,c,p,t,xi] for c in C for p in P) <= Capacity_W * y[w,t]))
            for c in C:
                model.addConstraint(xp.constraint(xp.Sum(x[w,c,t,xi] for w in W) == 1))

    for xi in Phi:
        for t in T:
            for c in C:
                for p in P:
                    Demand = agg_dem_cus_period_clus_scene_df.loc[(c,p,t),xi]
                    for w in W:
                        model.addConstraint(xp.constraint(v[w,c,p,t,xi] >= Demand * x[w,c,t,xi]))

    for xi in Phi:
        for t in T:
            for w in W:
                for p in P:
                    model.addConstraint(xp.constraint(xp.Sum(z[w,s,t] for s in S_P_dict[p]) >= xp.Sum(v[w,c,p,t,xi] for c in C)))
                    
    for s in S:
        Capacity_S = supplier_df.loc[s,'SupplierCapacity']
        for t in T:
            model.addConstraint(xp.constraint(xp.Sum(z[w,s,t] for w in W) <= Capacity_S))

    Setup_cost = xp.Sum(candidate_df.loc[w,'Setup'] * o[w] for w in W)
    Operating_cost = xp.Sum(candidate_df.loc[w,'Operating'] * y[w,t] for w in W for t in T)
    Tra_w_s_cost = xp.Sum(cost_w_to_s.loc[w,s] * z[w,s,t] for w in W for s in S for t in T)

    equal_prob = 1/len(agg_dem_cus_period_clus_scene_df.columns)
    Recourse = 0

    for xi in Phi:
        Tra_w_c_cost = xp.Sum(cost_w_to_cluster.loc[w,c] * v[w,c,p,t,xi] for w in W for c in C for p in P for t in T)
        Recourse += equal_prob * (Tra_w_c_cost)

    obj = Setup_cost + Operating_cost + Tra_w_s_cost + Recourse
    model.setObjective(obj, sense = xp.minimize)

    model.setControl('miprelstop', 1e-3)
    model.setControl('maxtime', time_limit_s)
    tic_time = time.time()
    # Solve the problem
    model.solve()
    toc_time = time.time()
    solve_time = toc_time - tic_time
    obj_value = model.getObjVal()

    mip_gap_percent = 100*(obj_value - model.getAttrib('bestbound'))/obj_value
    print(f'Solving Time: {solve_time}')
    print(f'Objective Value: {obj_value}')
    print(f'%Gaps: {mip_gap_percent}')

    x_matrix = model.getSolution(x)
    y_matrix = model.getSolution(y)
    v_matrix = model.getSolution(v)
    z_matrix = model.getSolution(z)

    return solve_time, obj_value, mip_gap_percent, x_matrix, y_matrix, v_matrix, z_matrix
