import math
import os
import pandas as pd
import numpy as np

from utils.parameters import Parameters


class ATCS(Parameters):
    def __init__(self, seed: int = 100):
        super().__init__()

        # Set the random seed for reproducibility
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        # Initialization the given data
        # Load robot data
        self.r_df = pd.read_csv('range.csv', index_col=0)  # Deterministic range
        self.r_s_df = pd.read_csv('range_scenarios.csv', index_col=0)  # Stochastic scenarios
        self.l_df = pd.read_csv('robot_locations.csv', index_col=0)  # Robot locations

        self.distance_matrix = self.get_distance_matrix()  # distance matrix between each robot

        # Stochastic values
        self.cc_df = self.determine_charging_decision()  # same size as range_sc_df. value 0 or 1 -> charge or not
        self.expected_range = self.calculate_stochastic_expected_range()  # expected range from range_sc_df

        # Initialize subset variables
        self.r_sub_df = None
        self.r_s_sub_df = None
        self.l_sub_df = None
        self.distance_matrix_sub = None
        self.cc_sub_df = None
        self.expected_range_sub = None

    def set_output_folder(self, solver_type: str, model_type: str, name: str):
        folder_name = f'{solver_type}_output/{model_type}/{name}'
        os.makedirs(folder_name , exist_ok=True)

    def choose_subset_point(self, n_sample: int = 100, randomized:bool = True):  # Generate Subset Data for MINLP Model
        if randomized:
            # Set random_state for reproducibility
            self.l_sub_df = self.l_df.sample(n_sample, random_state=self.seed)
        else:
            self.l_sub_df = self.l_df.head(n_sample)

        self.r_sub_df = self.r_df.loc[self.l_sub_df.index, :]
        self.r_s_sub_df = self.r_s_df.loc[self.l_sub_df.index, :]
        self.cc_sub_df = self.cc_df.loc[self.l_sub_df.index, :]
        self.expected_range_sub = self.expected_range.loc[self.l_sub_df.index, :]
        self.distance_matrix_sub = self.get_distance_matrix(sample_subset=True)
        
    def get_distance_matrix(self, sample_subset=False) -> np.ndarray:
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

    def determine_charging_decision(self) -> pd.DataFrame:
        prob_mat = np.exp(-(self.ld**2) * ((self.r_s_df - self.r_min)**2))
        uniform_mat = self.rng.uniform(low=0, high=1, size=len(self.r_s_df.to_numpy().flatten()))
        uniform_mat = uniform_mat.reshape(len(self.r_s_df), len(self.r_s_df.columns))
        should_charge_mat = uniform_mat <= prob_mat
        should_charge_mat = should_charge_mat.astype('int')
        
        return should_charge_mat

    def calculate_stochastic_expected_range(self) -> pd.DataFrame:
        return (self.cc_df * self.r_s_df).replace(0, np.NaN).mean(axis=1).to_frame(name="expected_range")
