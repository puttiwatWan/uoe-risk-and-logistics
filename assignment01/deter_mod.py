import preprocessing as pp
from model import DeterModel


def prepare_deter_input_data(n_clusters: int):
    # Read and Preprocess the data
    (cus_df, cand_df, sup_df, vehicle_df, distance_w_to_s_df, distance_w_to_c_df, demand_cus_period_df,
     demand_cus_period_scene_df) = pp.read_and_prep_data()

    # Customer Clustering
    cus_df, clust_center_df = pp.const_cluster_by_cus_loc(cus_df, n_clusters=n_clusters,
                                                          size_min=1, size_max=20, random_state=42)

    agg_dem_cus_period_df = pp.agg_dem_cus_period(demand_cus_period_df, cus_df)
    distance_w_to_cluster_df = pp.create_dis_mat_df(cand_df, clust_center_df, 'cityblock')

    # Create Cost
    tra_cost_w_to_cluster = pp.calculate_cost_from_w_to_cluster(distance_w_to_cluster_df, vehicle_df)
    tra_cost_w_to_s = pp.calculate_cost_from_w_to_s(distance_w_to_s_df, vehicle_df, sup_df)
    return agg_dem_cus_period_df, cand_df, sup_df, clust_center_df, tra_cost_w_to_s, tra_cost_w_to_cluster


def main():
    mod = DeterModel(*prepare_deter_input_data(50))
    mod.run(3600)


if __name__ == "__main__":
    main()
