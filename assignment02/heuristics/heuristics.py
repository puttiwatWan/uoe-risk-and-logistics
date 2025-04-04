import functools
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from config import config
from utils import time_spent_decorator, find_centroid


class HeuristicsResults:
    def __init__(self, objective_value: float = None,
                 stations_alloc: List[List[int]] = None,
                 stations_loc: List[Tuple[float, float]] = None,
                 ):
        self.objective_value = objective_value
        self.stations = stations_alloc.copy()
        self.stations_loc = stations_loc.copy()


class HeuristicSolver(ABC):
    def __init__(self, robot_loc: np.array(List[Tuple[float, float]]),
                 robot_range: np.array(List[float]),
                 robot_distance_matrix: np.ndarray,
                 ):
        # Given parameters
        self.m = 8  # Max chargers per station
        self.q = 2  # Max robots per charger

        self.c_b = 5000  # Build cost per station
        self.c_h = 1000  # Cost of moving a robot when out of range
        self.c_m = 500  # Cost per charger
        self.c_c = 0.42  # Charging cost per km
        self.ld = 0.012  # Lambda parameter for exponential distribution
        self.r_min = 10  # Minimum range of a robot
        self.r_max = 175  # Maximum range of a robot

        self.total_robots = len(robot_loc)
        self.robot_loc = robot_loc.copy()
        self.robot_range = robot_range.copy()

        self.original_robot_dist_matrix = robot_distance_matrix.copy()
        self.robot_dist_matrix = robot_distance_matrix.copy()
        self.stations: List[List[int]] = []

    def reset_solver(self):
        self.robot_dist_matrix = self.original_robot_dist_matrix.copy()
        self.stations = []

    def contains_penalty(self, station: Union[List[float] | Tuple[float, float]], centroid: Tuple) -> bool:
        for v in station:
            dis = math.dist(self.robot_loc[v], centroid)
            if dis > self.robot_range[v]:
                return True
        return False

    @functools.lru_cache(maxsize=256)
    def find_weighted_centroid(self, station: Tuple) -> tuple[float, float]:
        total_weight = sum((self.r_max - self.robot_range[v]) for v in station)
        x = sum((self.r_max - self.robot_range[v]) * self.robot_loc[v][0] for v in station) / total_weight
        y = sum((self.r_max - self.robot_range[v]) * self.robot_loc[v][1] for v in station) / total_weight
        return x, y

    @functools.lru_cache(maxsize=512)
    def find_cost_for_a_station(self, station: Tuple, centroid: Tuple) -> float:
        cost = math.ceil(len(station)/self.q) * self.c_m
        for v in station:
            dis = math.dist(self.robot_loc[v], centroid)
            if dis > self.robot_range[v]:
                cost += self.c_h + self.c_c * (self.r_max - self.robot_range[v])
            else:
                cost += self.c_c * (self.r_max - self.robot_range[v] + dis)
        return cost

    def find_total_cost(self, stations: List[List[int]], centroids: List[tuple[float, float]]) -> float:
        cost = 0
        for s, station in enumerate(stations):
            cost += self.find_cost_for_a_station(tuple(station), tuple(centroids[s]))

        cost += self.c_b * len(stations)
        return cost

    @abstractmethod
    def solve(self, **kwargs):
        ...

    @abstractmethod
    def print_results(self):
        ...

    @abstractmethod
    def get_heuristics_results(self) -> HeuristicsResults:
        ...


class ConstructionHeuristicSolver(HeuristicSolver):
    def __init__(self, robot_loc: np.array(List[Tuple[float, float]]),
                 robot_range: np.array(List[float]),
                 robot_distance_matrix: np.ndarray,
                 ):
        super().__init__(robot_loc=robot_loc,
                         robot_range=robot_range,
                         robot_distance_matrix=robot_distance_matrix)

    def find_next_robot(self, robot: int) -> int:
        if len(np.nonzero(self.robot_dist_matrix.flatten())[0]) == 0:
            return -1
        next_robot = np.where(self.robot_dist_matrix[robot] ==
                              np.min(self.robot_dist_matrix[robot, np.nonzero(self.robot_dist_matrix[robot])]))[0][0]
        self.robot_dist_matrix[:, next_robot] = 0
        return int(next_robot)

    @time_spent_decorator
    def solve(self, starting_robot: int = 0):
        # initialize a station with the starting robot
        self.robot_dist_matrix[:, starting_robot] = 0
        self.stations.append([starting_robot])

        robot = self.find_next_robot(starting_robot)
        centroids = [self.find_weighted_centroid(tuple(station)) for station in self.stations]
        prev_cost = self.find_total_cost(self.stations, centroids)

        while robot != -1:
            # try adding a new station
            new_station = [robot]
            tmp_stations = self.stations.copy()
            tmp_stations.append(new_station)
            centroids = [self.find_weighted_centroid(tuple(station)) for station in tmp_stations]
            cost = self.find_total_cost(tmp_stations, centroids)  # keep the cost to compare
            station_to_join = len(self.stations)  # need to add new index

            # compare with when adding to each station
            for s, station in enumerate(self.stations):
                # if the station is full -> skip
                if len(station) >= self.m * self.q:
                    continue

                # if contains out of range -> skip
                tmp_station = station + [robot]
                centroid = self.find_weighted_centroid(tuple(station))
                if self.contains_penalty(tmp_station, centroid):
                    continue

                tmp_stations = self.stations.copy()  # copy current stations
                tmp_stations[s] = tmp_station  # add the robot to the s station
                centroids = [self.find_weighted_centroid(tuple(station)) for station in tmp_stations]
                tmp_cost = self.find_total_cost(tmp_stations, centroids)

                # keep the current station if it decreases the cost
                if tmp_cost - prev_cost < cost - prev_cost:
                    cost = tmp_cost
                    station_to_join = s

            try:
                # add the robot to the chosen station
                self.stations[station_to_join].append(robot)
            except IndexError:
                # if a new station needs to be created
                self.stations.append([robot])

            prev_cost = cost
            robot = self.find_next_robot(robot)

    def print_results(self):
        centroids = [self.find_weighted_centroid(tuple(station)) for station in self.stations]
        print(f"The total cost is {self.find_total_cost(self.stations, centroids)}")
        print(f"robot in stations: {self.stations}")

        s_loc = []
        for station in self.stations:
            s_loc.append(self.find_weighted_centroid(tuple(station)))
        print(f"stations location: {s_loc}")

    def get_heuristics_results(self) -> HeuristicsResults:
        s_loc = []
        for station in self.stations:
            s_loc.append(self.find_weighted_centroid(tuple(station)))
        return HeuristicsResults(objective_value=self.find_total_cost(self.stations, s_loc),
                                 stations_loc=s_loc,
                                 stations_alloc=self.stations.copy())


class ImprovementCentroidHeuristics(HeuristicSolver):
    def __init__(self, robot_loc: np.array(List[Tuple[float, float]]),
                 robot_range: np.array(List[float]),
                 robot_distance_matrix: np.ndarray,
                 results: HeuristicsResults
                 ):
        super().__init__(robot_loc=robot_loc,
                         robot_range=robot_range,
                         robot_distance_matrix=robot_distance_matrix)

        self.stations = results.stations.copy()

        # Calculate current stations locations and improving direction
        self.target_stations_loc = []
        self.stations_loc = results.stations_loc.copy()
        for station in results.stations:
            centroid = find_centroid(robot_loc[station])
            self.target_stations_loc.append(centroid)

        # a small number to stop improving if the improvement is less than this number
        self.epsilon = config.improvement_centroid.epsilon

    @time_spent_decorator
    def solve(self, epsilon=config.improvement_centroid.epsilon):
        print(f"Initial target location: {self.target_stations_loc}")
        self.epsilon = epsilon

        for s, station in enumerate(self.stations):
            improved_cost = self.epsilon
            penalty_count = 0
            while improved_cost >= self.epsilon:
                if penalty_count > config.improvement_centroid.skip_after_penalty_count:
                    break

                new_centroid = find_centroid([self.stations_loc[s], self.target_stations_loc[s]])
                if self.contains_penalty(station, new_centroid):
                    # if penalty incurs, go to next iteration and calculate new_centroid using the one causing penalty
                    self.target_stations_loc[s] = new_centroid
                    penalty_count += 1
                else:
                    # if no penalty, update to new location and calculate improved_cost
                    old_cost = self.find_cost_for_a_station(tuple(station), tuple(self.stations_loc[s]))
                    new_cost = self.find_cost_for_a_station(tuple(station), new_centroid)
                    if new_cost >= old_cost:
                        break

                    improved_cost = old_cost - new_cost  # expected to be negative
                    self.stations_loc[s] = new_centroid
                    penalty_count = 0

    def print_results(self):
        print(f"The improved cost is {self.find_total_cost(self.stations, self.stations_loc)}")
        print(f"stations location: {self.stations_loc}")

    def get_heuristics_results(self) -> HeuristicsResults:
        return HeuristicsResults(objective_value=self.find_total_cost(self.stations, self.stations_loc),
                                 stations_loc=self.stations_loc.copy(),
                                 stations_alloc=self.stations.copy())


class ImprovementReduceStationHeuristics(HeuristicSolver):
    def solve(self, **kwargs):
        pass

    def print_results(self):
        pass

    def get_heuristics_results(self) -> HeuristicsResults:
        pass
