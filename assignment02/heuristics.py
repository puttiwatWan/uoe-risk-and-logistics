import functools
import math
from typing import List, Tuple

import numpy as np

from atcs import ATCS
from utils import time_spent_decorator


class HeuristicSolver:
    def __init__(self, robot_loc: List[Tuple[float, float]],
                 robot_range: List[float],
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
        self.robot_loc = np.array(robot_loc)
        self.robot_range = np.array(robot_range)
        self.original_robot_dist_matrix = robot_distance_matrix.copy()

        self.robot_dist_matrix = robot_distance_matrix.copy()
        self.stations: List[List[int]] = []

    @time_spent_decorator
    def __reset_solver(self):
        self.robot_dist_matrix = self.original_robot_dist_matrix.copy()
        self.stations = []

    @time_spent_decorator
    def __contains_out_of_range(self, station: List[int]) -> bool:
        centroid = self.find_weighted_centroid(tuple(station))
        for v in station:
            dis = math.dist(self.robot_loc[v], centroid)
            if dis > self.robot_range[v]:
                return True
        return False

    @time_spent_decorator
    @functools.lru_cache(maxsize=1024)
    def find_weighted_centroid(self, robots: Tuple) -> tuple[float, float]:
        total_weight = sum(self.r_max - self.robot_range[v] for v in robots)
        x = sum(self.r_max - self.robot_range[v] * self.robot_loc[v][0] for v in robots)/total_weight
        y = sum(self.r_max - self.robot_range[v] * self.robot_loc[v][1] for v in robots)/total_weight
        return x, y

    @time_spent_decorator
    @functools.lru_cache(maxsize=1024)
    def find_cost_for_a_station(self, station: Tuple) -> float:
        cost = math.ceil(len(station)/self.q) * self.c_m
        centroid = self.find_weighted_centroid(station)
        for v in station:
            dis = math.dist(self.robot_loc[v], centroid)
            if dis > self.robot_range[v]:
                cost += self.c_h + self.c_c * (self.r_max - self.robot_range[v])
            else:
                cost += self.c_c * (self.r_max - self.robot_range[v] + dis)
        return cost

    @time_spent_decorator
    def find_total_cost(self, stations: List[List[int]]) -> float:
        cost = 0
        for station in stations:
            cost += self.find_cost_for_a_station(tuple(station))

        cost += self.c_b * len(stations)
        return cost

    @time_spent_decorator
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
        prev_cost = self.find_total_cost(self.stations)

        while robot != -1:
            # try adding a new station
            new_station = [robot]
            tmp_stations = self.stations.copy()
            tmp_stations.append(new_station)
            cost = self.find_total_cost(tmp_stations)  # keep the cost to compare
            station_to_join = len(self.stations)  # need to add new index

            # compare with when adding to each station
            for s, station in enumerate(self.stations):
                # if the station is full -> skip
                if len(station) >= self.m * self.q:
                    continue

                # if contains out of range -> skip
                tmp_station = station + [robot]
                if self.__contains_out_of_range(tmp_station):
                    continue

                tmp_stations = self.stations.copy()  # copy current stations
                tmp_stations[s] = tmp_station  # add the robot to the s station
                tmp_cost = self.find_total_cost(tmp_stations)

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

    @time_spent_decorator
    def print_results(self):
        print(f"The total cost is {self.find_total_cost(self.stations)}")
        print(self.stations)


@time_spent_decorator
def main():
    data = ATCS(seed=1)

    robot_loc = data.l_df.to_numpy()
    robot_range = data.r_df.to_numpy().flatten()
    dist_matrix = data.get_distance_matrix()
    solver = HeuristicSolver(robot_range=robot_range,
                             robot_loc=robot_loc,
                             robot_distance_matrix=dist_matrix)
    solver.solve()
    solver.print_results()


if __name__ == "__main__":
    main()
