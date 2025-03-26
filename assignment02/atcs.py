import math
import os
import pandas as pd
import numpy as np


class ATCS:
    def __init__(self, seed: int = 100):
        # Set the random seed for reproducibility
        self.seed = seed
        np.random.seed(self.seed) 

        # Initialization the given data
        # Load robot data
        self.r_df = pd.read_csv('range.csv', index_col=0)  # Deterministic range
        self.r_s_df = pd.read_csv('range_scenarios.csv', index_col=0)  # Stochastic scenarios
        self.l_df = pd.read_csv('robot_locations.csv', index_col=0)  # Robot locations
        
        # Given parameters
        self.m = 8  # Max chargers per station
        self.q = 2  # Max robots per charger
        self.c_b = 5000  # Investment cost per station
        self.c_h = 1000  # Cost of moving a robot
        self.c_m = 500  # Maintenance cost per charger
        self.c_c = 0.42  # Charging cost per km
        self.ld = 0.012  # Lambda parameter for exponential distribution
        self.r_min = 10  # Minimum range of a robot
        self.r_max = 175  # Maximum range of a robot

    def set_output_folder(self, solver_type:str, model_type:str, name:str):
        folder_name = f'{solver_type}_output/{model_type}/{name}'
        os.makedirs(folder_name , exist_ok=True)

    def choose_subset_point(self, n_sample: int = 100, random_state: int = 100):  # Generate Subset Data for MINLP Model
        # Set random_state for reproducibility
        self.l_sub_df = self.l_df.sample(n_sample, random_state=self.seed).copy()
        self.r_sub_df = self.r_df.loc[self.l_sub_df.index,:].copy()
        self.r_s_sub_df = self.r_s_df.loc[self.l_sub_df.index, :].copy()

    def get_distance_matrix(self, sample_subset = False) -> np.ndarray:
        if sample_subset:
            loc = self.l_sub_df.to_numpy()
        else:
            loc = self.l_df.to_numpy()
        dist_mat = np.zeros((len(loc), len(loc)))
        for i in range(len(loc)):
            for j in range(i, len(loc), 1):
                dist_mat[i, j] = math.dist(loc[i], loc[j])
                dist_mat[j, i] = dist_mat[i, j]

        return dist_mat

    # def compute_charging_probability(self, r_i): # Not Sure
        # return np.exp(-self.ld**2 * (r_i - self.r_min) ** 2)
