import pandas as pd
import numpy as np
import read_data as rdt
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

def dis_mat_from_dict(dis_dict:dict, mat_size: tuple)->pd.DataFrame:
    # Distance matrix from supplier to warehouse
    dis_mat = np.zeros(mat_size)
    j = 0
    for k,i in dis_dict.items():
        dis_mat[:,j] = i
        j+=1
    dis_mat_df = pd.DataFrame(dis_mat.T)

    return dis_mat_df

    
def read_and_prep_data()->tuple:
    dat_dict = rdt.parse_dat_file('CaseStudyData.txt')

    nbCustomers = int(dat_dict['nbCustomers'][0])
    nbCandidates = int(dat_dict['nbCandidates'][0])
    nbSuppliers = int(dat_dict['nbSuppliers'][0])
    nbProductGroups = int(dat_dict['nbProductGroups'][0])
    nbVehicleTypes = int(dat_dict['nbVehicleTypes'][0])
    nbPeriods = int(dat_dict['nbPeriods'][0])
    nbScenarios = int(dat_dict['nbScenarios'][0])


    CustomerId = dat_dict['CustomerId']
    CustomerEasting = dat_dict['CustomerEasting']
    CustomerNorthing = dat_dict['CustomerNorthing']
    CustomerPopulation = dat_dict['CustomerPopulation']

    CandidateId = dat_dict['CandidateId']
    CandidateEasting = dat_dict['CandidateEasting']
    CandidateNorthing = dat_dict['CandidateNorthing']

    SupplierId = dat_dict['SupplierId']
    SupplierEasting = dat_dict['SupplierEasting']
    SupplierNorthing = dat_dict['SupplierNorthing']
    SupplierProductGroup = dat_dict['SupplierProductGroup']
    SupplierCapacity = dat_dict['SupplierCapacity']
    SupplierVehicleType = dat_dict['SupplierVehicleType']

    VehicleCapacity = dat_dict['VehicleCapacity']
    VehicleCostPerMileOverall = dat_dict['VehicleCostPerMileOverall']
    VehicleCostPerMileAndTonneOverall = dat_dict['VehicleCostPerMileAndTonneOverall']
    VehicleCO2PerMileAndTonne = dat_dict['VehicleCO2PerMileAndTonne']

    Setup = dat_dict['Setup']
    Operating = dat_dict['Operating']
    Capacity = dat_dict['Capacity']

    CustomerDemand = dat_dict['CustomerDemand']
    CustomerDemandPeriods = dat_dict['CustomerDemandPeriods']
    CustomerDemandPeriodScenarios = dat_dict['CustomerDemandPeriodScenarios']

    DistanceCandidateSupplier = dat_dict['DistanceCandidateSupplier']
    DistanceCandidateCustomer = dat_dict['DistanceCandidateCustomer']



    # Preprocessed into dataframe
    customer_df = pd.DataFrame({        'CustomerId': CustomerId, 'CustomerEasting':CustomerEasting,
                                    'CustomerNorthing':CustomerNorthing, 'CustomerPopulation': CustomerPopulation})

    candidate_df = pd.DataFrame(    {   'CandidateId':CandidateId, 'CandidateEasting':CandidateEasting, 'CandidateNorthing':CandidateNorthing, 'Capacity':Capacity[(1,)],
                                        'Setup':Setup[(1,)], 'Operating':Operating[(1,)]})

    supplier_df = pd.DataFrame(     {   'SupplierId':SupplierId, 'SupplierEasting':SupplierEasting, 'SupplierNorthing':SupplierNorthing,
                                        'SupplierProductGroup':SupplierProductGroup, 'SupplierCapacity':SupplierCapacity[(1,)], 'SupplierVehicleType':SupplierVehicleType})

    vehicle_df = pd.DataFrame(      {   'VehicleCapacity':VehicleCapacity, 'VehicleCostPerMileOverall':VehicleCostPerMileOverall, 
                                        'VehicleCostPerMileAndTonneOverall':VehicleCostPerMileAndTonneOverall, 'VehicleCO2PerMileAndTonne':VehicleCO2PerMileAndTonne})

    demand_cus_period_df = pd.DataFrame(CustomerDemandPeriods).T
    demand_cus_period_df.reset_index(inplace=True)
    demand_cus_period_df.drop('level_2', axis = 1, inplace=True)
    demand_cus_period_df.rename(columns={'level_0':'CustomerIndex', 'level_1':'ProductIndex'}, inplace=True)
    demand_cus_period_df['CustomerIndex'] = demand_cus_period_df['CustomerIndex'] - 1 # set the first index into 0
    demand_cus_period_df.set_index(['CustomerIndex','ProductIndex'], inplace= True)

    demand_cus_period_scene_df = pd.DataFrame(CustomerDemandPeriodScenarios).T
    demand_cus_period_scene_df.reset_index(inplace=True)
    demand_cus_period_scene_df.rename(columns={'level_0':'CustomerIndex', 'level_1':'ProductIndex', 'level_2':'PeriodIndex'}, inplace=True)
    demand_cus_period_scene_df.drop('level_3',axis=1, inplace=True)
    demand_cus_period_scene_df['CustomerIndex'] = demand_cus_period_scene_df['CustomerIndex'] - 1 # set the first index into 0
    demand_cus_period_scene_df.set_index(['CustomerIndex','ProductIndex','PeriodIndex'], inplace= True)

    # Default Distance Matrix before Aggregation
    distance_w_to_s_df = dis_mat_from_dict(DistanceCandidateSupplier, (nbSuppliers, nbCandidates))
    distance_w_to_c_df = dis_mat_from_dict(DistanceCandidateCustomer, (nbCustomers, nbCandidates))

    return customer_df, candidate_df, supplier_df, vehicle_df, distance_w_to_s_df, distance_w_to_c_df, demand_cus_period_df, demand_cus_period_scene_df


def ori_cluster_by_cus_loc(customer_df:pd.DataFrame, n_clusters:int=100, random_state:int=42, n_init:int=10)->tuple:
    df = customer_df.copy()
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(customer_df[['CustomerEasting','CustomerNorthing','CustomerPopulation']])    

    cluster_center_df = pd.DataFrame(kmeans.cluster_centers_)
    cluster_center_df.drop(2,axis = 1, inplace = True)
    cluster_center_df.reset_index(inplace=True)
    cluster_center_df.rename(columns = {'index':'Cluster', 0:'CustomerEasting', 1:'CustomerNorthing'}, inplace = True)
    cluster_center_df.set_index('Cluster',inplace=True)

    return df, cluster_center_df

def const_cluster_by_cus_loc(customer_df:pd.DataFrame, n_clusters:int=100, size_min:int=4, size_max:int=400, random_state:int=42)->tuple:
    df = customer_df.copy()

    kmeans_constrained = KMeansConstrained(
                                            n_clusters=n_clusters,
                                            size_min=size_min,
                                            size_max=size_max,
                                            random_state=random_state
                                        )

    df['Cluster'] = kmeans_constrained.fit_predict(customer_df[['CustomerEasting','CustomerNorthing','CustomerPopulation']])

    cluster_center_df = pd.DataFrame(kmeans_constrained.cluster_centers_)
    cluster_center_df.drop(2,axis = 1, inplace = True)
    cluster_center_df.reset_index(inplace=True)
    cluster_center_df.rename(columns = {'index':'Cluster', 0:'CustomerEasting', 1:'CustomerNorthing'}, inplace = True)
    cluster_center_df.set_index('Cluster',inplace=True)

    return df, cluster_center_df

def agg_dem_cus_period(demand_cus_period_df:pd.DataFrame, customer_df_with_clus:pd.DataFrame)->pd.DataFrame:
    df = demand_cus_period_df.copy()
    df = df.reset_index().merge(customer_df_with_clus.reset_index()[['index','Cluster']], left_on='CustomerIndex', right_on='index', how = 'left')
    df.drop(['CustomerIndex','index'], axis =1 ,inplace=True)
    df_group = df.groupby(['Cluster','ProductIndex']).sum()

    return df_group

def agg_dem_cus_period_scene(demand_cus_period_scene_df:pd.DataFrame, customer_df_with_clus:pd.DataFrame)->pd.DataFrame:
    df = demand_cus_period_scene_df.copy()
    df = df.reset_index().merge(customer_df_with_clus.reset_index()[['index','Cluster']], left_on='CustomerIndex', right_on='index', how = 'left')
    df.drop(['CustomerIndex','index'], axis =1 ,inplace=True)
    df_group = df.groupby(['Cluster','ProductIndex','PeriodIndex']).sum()

    return df_group