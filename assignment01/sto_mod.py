import preprocessing as pp
import numpy as np
from model import StochasticModel


def prepare_sto_input_data(n_cus_clusters: int, n_scene_clusters: int = 3):
    # Read and Preprocess the data
    (cus_df, cand_df, sup_df, vehicle_df, distance_w_to_s_df, _, _,
     demand_cus_period_scene_df) = pp.read_and_prep_data()

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
    return agg_dem_cus_period_clus_scene_df, cand_df, sup_df, clust_center_df, tra_cost_w_to_s, tra_cost_w_to_cluster


def main():
    mod = StochasticModel(*prepare_sto_input_data(50, 3))
    mod.run(3600)


if __name__ == "__main__":
    main()
