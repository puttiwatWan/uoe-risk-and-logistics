import pandas as pd
import numpy as np
import read_data as rdt

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


    distance_w_to_s_df = dis_mat_from_dict(DistanceCandidateSupplier, (nbSuppliers, nbCandidates))
    distance_w_to_c_df = dis_mat_from_dict(DistanceCandidateCustomer, (nbCustomers, nbCandidates))

    return customer_df, candidate_df, supplier_df, vehicle_df, distance_w_to_s_df, distance_w_to_c_df, CustomerDemand, CustomerDemandPeriods, CustomerDemandPeriodScenarios
