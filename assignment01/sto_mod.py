import preprocessing as pp
import pandas as pd
import numpy as np
import time
import xpress as xp  # FICO Xpress Solver


xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')


def sto_model(agg_dem_cus_period_clus_scene_df: pd.DataFrame,
              candidate_df: pd.DataFrame,
              supplier_df: pd.DataFrame,
              cluster_center_df: pd.DataFrame,
              cost_w_to_s: pd.DataFrame,
              cost_w_to_cluster: pd.DataFrame,
              time_limit_s: int = 3600):
    model = xp.problem(name='MEWLP_Stochastics')
    # Sets and Notation
    S = list(supplier_df.index)  # Supplier Index
    S_P0 = list(supplier_df[supplier_df['SupplierProductGroup'] == 0]['SupplierProductGroup'].index)
    S_P1 = list(supplier_df[supplier_df['SupplierProductGroup'] == 1]['SupplierProductGroup'].index)
    S_P2 = list(supplier_df[supplier_df['SupplierProductGroup'] == 2]['SupplierProductGroup'].index)
    S_P3 = list(supplier_df[supplier_df['SupplierProductGroup'] == 3]['SupplierProductGroup'].index)
    S_P_dict = {0: S_P0, 1: S_P1, 2: S_P2, 3: S_P3}

    W = list(candidate_df.index)  # Warehouse Index
    C = list(cluster_center_df.index)  # Cluster Index
    P = list(agg_dem_cus_period_clus_scene_df.reset_index()['ProductIndex'].unique())
    T = list(agg_dem_cus_period_clus_scene_df.reset_index()['PeriodIndex'].unique())
    Phi = list(agg_dem_cus_period_clus_scene_df.columns)

    # Output Variables
    x = model.addVariables(W, C, T, Phi, name="x", vartype=xp.binary)
    y = model.addVariables(W, name="y", vartype=xp.binary)
    o = model.addVariables(W, T, name="o", vartype=xp.binary)
    z = model.addVariables(W, S, T, name="z", vartype=xp.continuous)

    # Constraints
    model.addConstraint(y[w] >= o[w, t] for w in W for t in T)
    model.addConstraint(o[w, t] >= o[w, t - 1] for w in W for t in T if t != 0)
    model.addConstraint(x[w, c, t, xi] <= o[w, t] for w in W for t in T for c in C for xi in Phi)

    # Limit warehouse capacity >= total covered demand
    model.addConstraint(xp.Sum(agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * x[w, c, t, xi]
                               for c in C for p in P)
                        <= candidate_df.loc[w, 'Capacity'] * o[w, t]
                        for w in W for t in T for xi in Phi)

    # A customer needs to be covered by a warehouse
    model.addConstraint(xp.Sum(x[w, c, t, xi] for w in W) == 1 for c in C for t in T for xi in Phi)

    model.addConstraint(xp.Sum(z[w, s, t] for s in S_P_dict[p]) >=
                        xp.Sum(agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * x[w, c, t, xi] for c in C)
                        for w in W for p in P for t in T for xi in Phi)

    model.addConstraint(xp.Sum(z[w, s, t] for w in W) <= supplier_df.loc[s, 'SupplierCapacity'] for s in S for t in T)

    Setup_cost = xp.Sum(candidate_df.loc[w, 'Setup'] * y[w] for w in W)
    Operating_cost = xp.Sum(candidate_df.loc[w, 'Operating'] * o[w, t] for w in W for t in T)
    Tra_w_s_cost = xp.Sum(cost_w_to_s.loc[w, s] * z[w, s, t] for w in W for s in S for t in T)

    equal_prob = 1 / len(agg_dem_cus_period_clus_scene_df.columns)
    Recourse = xp.Sum(equal_prob * xp.Sum(cost_w_to_cluster.loc[w, c] *
                                          agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * x[w, c, t, xi]
                                          for w in W for c in C for p in P for t in T)
                      for xi in Phi)

    obj = Setup_cost + Operating_cost + Tra_w_s_cost + Recourse
    model.setObjective(obj, sense=xp.minimize)

    model.setControl('miprelstop', 1e-3)
    model.setControl('maxtime', time_limit_s)
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
    z_sol = model.getSolution(z)

    def print_results():
        print(f'Solving Time: {solve_time}')
        print(f'Objective Value: {objective_value}')
        print(f'%Gaps: {mip_gap_percentage}')

        set_cost_sol = sum(candidate_df.loc[w, 'Setup'] * y_sol[w] for w in W)
        operation_cost_sol = sum(candidate_df.loc[w, 'Operating'] * o_sol[w, t] for w in W for t in T)
        w_s_cost_sol = sum(cost_w_to_s.loc[w, s] * z_sol[w, s, t] for w in W for s in S for t in T)
        recourse_sol = sum(equal_prob * sum(cost_w_to_cluster.loc[w, c] *
                                            agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * x_sol[w, c, t, xi]
                                            for w in W for c in C for p in P for t in T)
                           for xi in Phi)

        print(f"Setup cost: {set_cost_sol}")
        print(f"Operation cost: {operation_cost_sol}")
        print(f"Transportation from Supplier to Warehouse cost: {w_s_cost_sol}")
        print(f"Recourse cost: {recourse_sol}")

        for t in T:
            opened_warehouse = [w for w in W if y_sol[w, t] == 1]
            print(f"Warehouse open in T: {t} are: {opened_warehouse}")

    print_results()

    return solve_time, objective_value, mip_gap_percentage, x_sol, y_sol, o_sol, z_sol


def main():
    # Read and Preprocess the data
    (cus_df, cand_df, sup_df, vehicle_df, distance_w_to_s_df, _, _,
     demand_cus_period_scene_df) = pp.read_and_prep_data()

    time_limit_s = 3600
    n_cus_clusters = 50
    n_scene_clusters = 3

    print(f'Now running Case N_cus_cluster = {n_cus_clusters}')
    print(f'Now running Case N_scene_cluster = {n_scene_clusters}')
    # Customer Clustering
    cus_df, clust_center_df = pp.const_cluster_by_cus_loc(cus_df, n_clusters=n_cus_clusters, size_min=1, size_max=20,
                                                          random_state=42)
    agg_dem_cus_period_scene_df = pp.agg_dem_cus_period_scene(demand_cus_period_scene_df, cus_df)
    distance_w_to_cluster_df = pp.create_dis_mat_df(cand_df, clust_center_df, 'cityblock')

    # Create Cost
    tra_cost_w_to_cluster = pp.calculate_cost_from_w_to_cluster(distance_w_to_cluster_df, vehicle_df)
    tra_cost_w_to_s = pp.calculate_cost_from_w_to_s(distance_w_to_s_df, vehicle_df, sup_df)

    # Scenarios Clustering
    _, scene_cluster_center_df = pp.constrained_kmeans_clustering(agg_dem_cus_period_scene_df,
                                                                  n_clusters=n_scene_clusters,
                                                                  size_min=np.floor(
                                                                      len(agg_dem_cus_period_scene_df.columns) /
                                                                      n_scene_clusters),
                                                                  size_max=len(agg_dem_cus_period_scene_df.columns),
                                                                  random_state=42)
    agg_dem_cus_period_clus_scene_df = pp.agg_scene_df(agg_dem_cus_period_scene_df, scene_cluster_center_df)

    sto_model(agg_dem_cus_period_clus_scene_df, cand_df, sup_df, clust_center_df, tra_cost_w_to_s,
              tra_cost_w_to_cluster, time_limit_s)


if __name__ == "__main__":
    main()
