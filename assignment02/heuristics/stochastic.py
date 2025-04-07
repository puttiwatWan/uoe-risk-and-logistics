import functools
import math
from abc import ABC
from typing import List, Tuple

import numpy as np

from .deterministic import HeuristicsResults, ConstructionHeuristicSolver, HeuristicSolver


class StochasticHeuristicSolver(HeuristicSolver, ABC):
    def __init__(self, robot_range_scenarios: np.array(List[List[float]]),
                 robot_scenario_used: np.array(List[List[int]]),
                 robot_loc: np.array(List[Tuple[float, float]]) = None,
                 robot_range: np.array(List[float]) = None,
                 robot_distance_matrix: np.ndarray = None):

        super().__init__(robot_loc, robot_range, robot_distance_matrix)

        self.robot_range_scenarios = robot_range_scenarios.copy()
        self.robot_scenario_used = robot_scenario_used.copy()
        self.n_scenarios = robot_range_scenarios.shape[1]

    @functools.lru_cache(maxsize=512)
    def find_cost_for_a_station(self, station: Tuple, centroid: Tuple, penalty_in_station: Tuple = None) -> float:
        total_cost = math.ceil(len(station) / self.q) * self.c_m  # initialize with chargers cost
        scenarios_costs = []
        for sc in range(self.n_scenarios):
            cost = 0
            for v in station:
                if not self.robot_scenario_used[v, sc]:
                    continue

                dis = math.dist(self.robot_loc[v], centroid)
                if dis > self.robot_range_scenarios[v, sc]:
                    cost += self.c_h + self.c_c * (self.r_max - self.robot_range_scenarios[v, sc])
                else:
                    cost += self.c_c * (self.r_max - self.robot_range_scenarios[v, sc] + dis)

            if penalty_in_station:
                for v in penalty_in_station:
                    cost += self.c_h + self.c_c * (self.r_max - self.robot_range_scenarios[v, sc])

            scenarios_costs.append(cost)
        total_cost += np.sum(np.array(scenarios_costs) * (1/self.n_scenarios))
        return total_cost


class StochasticConstructionHeuristicSolver(StochasticHeuristicSolver, ConstructionHeuristicSolver):
    def __init__(self, robot_loc: np.array(List[Tuple[float, float]]),
                 robot_expected_range: np.array(List[float]),
                 robot_distance_matrix: np.ndarray,
                 robot_range_scenarios: np.array(List[List[float]]),
                 robot_scenario_used: np.array(List[List[int]])
                 ):

        StochasticHeuristicSolver.__init__(self, robot_range_scenarios=robot_range_scenarios,
                                           robot_scenario_used=robot_scenario_used)
        ConstructionHeuristicSolver.__init__(self, robot_loc=robot_loc,
                                             robot_range=robot_expected_range,
                                             robot_distance_matrix=robot_distance_matrix)

    # def print_results(self):
    #     pass
    #
    # def get_heuristics_results(self) -> HeuristicsResults:
    #     pass
