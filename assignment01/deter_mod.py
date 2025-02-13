import preprocessing as pp
import pandas as pd
import time
import xpress as xp  # FICO Xpress Solver


xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')


def deter_model(aggregated_demand_period_df: pd.DataFrame,
                candidate_df: pd.DataFrame,
                supplier_df: pd.DataFrame,
                cluster_center_df: pd.DataFrame,
                cost_w_to_s: pd.DataFrame,
                cost_w_to_cluster: pd.DataFrame,
                time_limit_in_sec: int = 3600):
    # Create the model
    model = xp.problem(name='MEWLP_Deterministic')
    # Sets and Notation
    S = list(supplier_df.index)  # Supplier Index
    S_P0 = list(supplier_df[supplier_df['SupplierProductGroup'] == 0]['SupplierProductGroup'].index)
    S_P1 = list(supplier_df[supplier_df['SupplierProductGroup'] == 1]['SupplierProductGroup'].index)
    S_P2 = list(supplier_df[supplier_df['SupplierProductGroup'] == 2]['SupplierProductGroup'].index)
    S_P3 = list(supplier_df[supplier_df['SupplierProductGroup'] == 3]['SupplierProductGroup'].index)
    S_P_dict = {0: S_P0, 1: S_P1, 2: S_P2, 3: S_P3}

    W = list(candidate_df.index)  # Warehouse Index
    C = list(cluster_center_df.index)  # Cluster Index
    P = list(aggregated_demand_period_df.reset_index()['ProductIndex'].unique())
    T = list(list(aggregated_demand_period_df.columns))

    # Output Variables
    x = model.addVariables(W, C, T, name="x", vartype=xp.binary)
    o = model.addVariables(W, T, name="y", vartype=xp.binary)
    y = model.addVariables(W, name="o", vartype=xp.binary)
    v = model.addVariables(W, C, P, T, name="v", vartype=xp.continuous)
    z = model.addVariables(W, S, T, name="z", vartype=xp.continuous)

    # Constraints
    model.addConstraint(y[w] >= o[w, t] for w in W for t in T)
    model.addConstraint(xp.Sum(v[w, c, p, t] for c in C for p in P) <= candidate_df.loc[w, 'Capacity'] * o[w, t]
                        for w in W for t in T)
    model.addConstraint(o[w, t] >= o[w, t - 1] for w in W for t in T if t != 0)
    model.addConstraint(xp.Sum(z[w, s, t] for s in S_P_dict[p]) == xp.Sum(v[w, c, p, t] for c in C)
                        for w in W for t in T for p in P)
    model.addConstraint(x[w, c, t] <= o[w, t] for w in W for c in C for t in T)

    model.addConstraint(xp.Sum(x[w, c, t] for w in W) == 1 for c in C for t in T)
    model.addConstraint(v[w, c, p, t] == aggregated_demand_period_df.loc[(c, p), t] * x[w, c, t]
                        for w in W for c in C for t in T for p in P)
    model.addConstraint(xp.Sum(z[w, s, t] for w in W) <= supplier_df.loc[s, 'SupplierCapacity'] for s in S for t in T)

    Setup_cost = xp.Sum(candidate_df.loc[w, 'Setup'] * y[w] for w in W)
    Operating_cost = xp.Sum(candidate_df.loc[w, 'Operating'] * o[w, t] for w in W for t in T)
    Tra_w_s_cost = xp.Sum(cost_w_to_s.loc[w, s] * z[w, s, t] for w in W for s in S for t in T)
    Tra_w_c_cost = xp.Sum(cost_w_to_cluster.loc[w, c] * v[w, c, p, t] for w in W for c in C for p in P for t in T)

    obj = Setup_cost + Operating_cost + Tra_w_s_cost + Tra_w_c_cost

    model.setObjective(obj, sense=xp.minimize)
    model.setControl('miprelstop', 1e-3)
    model.setControl('maxtime', time_limit_in_sec)
    tic_time = time.time()
    # Solve the problem
    model.solve()
    toc_time = time.time()
    solve_time = toc_time - tic_time

    objective_value = model.getObjVal()
    mip_gap_percentage = 100 * (objective_value - model.getAttrib('bestbound')) / objective_value

    x_sol = model.getSolution(x)
    y_sol = model.getSolution(y)
    o_sol = model.getSolution(o)
    v_sol = model.getSolution(v)
    z_sol = model.getSolution(z)

    def print_results():
        print(f'Solving Time: {solve_time}')
        print(f'Objective Value: {objective_value}')
        print(f'%Gaps: {mip_gap_percentage}')

        set_cost_sol = sum(candidate_df.loc[w, 'Setup'] * y_sol[w] for w in W)
        operation_cost_sol = sum(candidate_df.loc[w, 'Operating'] * o_sol[w, t] for w in W for t in T)
        w_s_cost_sol = sum(cost_w_to_s.loc[w, s] * z_sol[w, s, t] for w in W for s in S for t in T)
        w_c_cost_sol = sum(cost_w_to_cluster.loc[w, c] * v_sol[w, c, p, t] for w in W for c in C for p in P for t in T)

        print(f"Setup cost: {set_cost_sol}")
        print(f"Operation cost: {operation_cost_sol}")
        print(f"Transportation from Supplier to Warehouse cost: {w_s_cost_sol}")
        print(f"Transportation from Warehouse to Customer cost: {w_c_cost_sol}")

    print_results()

    return solve_time, objective_value, mip_gap_percentage, x, y, o, v, z


def main():
    # Read and Preprocess the data
    (cus_df, cand_df, sup_df, vehicle_df, distance_w_to_s_df, distance_w_to_c_df, demand_cus_period_df,
     demand_cus_period_scene_df) = pp.read_and_prep_data()

    time_limit_s = 3600
    each_n_clusters = 50

    print(f'Now running Case N_cluster = {each_n_clusters}')
    # Customer Clustering
    cus_df, clust_center_df = pp.const_cluster_by_cus_loc(cus_df, n_clusters=each_n_clusters,
                                                          size_min=1, size_max=20, random_state=42)

    agg_dem_cus_period_df = pp.agg_dem_cus_period(demand_cus_period_df, cus_df)
    distance_w_to_cluster_df = pp.create_dis_mat_df(cand_df, clust_center_df, 'cityblock')

    # Create Cost
    tra_cost_w_to_cluster = pp.calculate_cost_from_w_to_cluster(distance_w_to_cluster_df, vehicle_df)
    tra_cost_w_to_s = pp.calculate_cost_from_w_to_s(distance_w_to_s_df, vehicle_df, sup_df)

    deter_model(agg_dem_cus_period_df, cand_df, sup_df, clust_center_df, tra_cost_w_to_s, tra_cost_w_to_cluster,
                time_limit_s)


if __name__ == "__main__":
    main()
