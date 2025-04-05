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
                 stations_with_penalty: List[List[int]] = None,
                 ):
        self.objective_value = objective_value
        self.stations = stations_alloc.copy()
        self.stations_loc = stations_loc.copy()

        self.stations_with_penalty = [[] for _ in range(len(self.stations))]
        if stations_with_penalty:
            self.stations_with_penalty = stations_with_penalty.copy()


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
    def find_cost_for_a_station(self, station: Tuple, centroid: Tuple, penalty_in_station: Tuple = None) -> float:
        cost = math.ceil(len(station) / self.q) * self.c_m

        combined_station = list(station)
        if penalty_in_station:
            combined_station += list(penalty_in_station)

        for v in combined_station:
            dis = math.dist(self.robot_loc[v], centroid)
            if dis > self.robot_range[v]:
                cost += self.c_h + self.c_c * (self.r_max - self.robot_range[v])
            else:
                cost += self.c_c * (self.r_max - self.robot_range[v] + dis)
        return cost

    def find_total_cost(self, stations: List[List[int]],
                        centroids: List[tuple[float, float]],
                        stations_with_penalty: List[List[int]] = None) -> float:
        cost = 0
        for s, station in enumerate(stations):
            cost += self.find_cost_for_a_station(tuple(station), tuple(centroids[s]),
                                                 tuple(stations_with_penalty[s]) if stations_with_penalty else None)

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
        print(f"Total stations: {len(self.stations)}")
        print(f"robot in stations: {self.stations}")
        print(f"stations location: {self.stations_loc}")

    def get_heuristics_results(self) -> HeuristicsResults:
        return HeuristicsResults(objective_value=self.find_total_cost(self.stations, self.stations_loc),
                                 stations_loc=self.stations_loc.copy(),
                                 stations_alloc=self.stations.copy())


class ImprovementStationsReductionHeuristics(HeuristicSolver):
    def __init__(self, robot_loc: np.array(List[Tuple[float, float]]),
                 robot_range: np.array(List[float]),
                 robot_distance_matrix: np.ndarray,
                 results: HeuristicsResults
                 ):
        super().__init__(robot_loc=robot_loc,
                         robot_range=robot_range,
                         robot_distance_matrix=robot_distance_matrix)

        self.stations = results.stations.copy()
        self.stations_loc = results.stations_loc.copy()
        self.stations_penalty = results.stations_with_penalty.copy()

        self.min_robots_req = [m for m in range(self.q * self.m) if self.c_b < m * self.c_h][0]

    def solve(self, **kwargs):
        max_rps = self.m * self.q
        station_to_remove = []
        number_of_stations = len(self.stations)
        for s in range(number_of_stations):
            # filter for only stations with <= 5 robots
            if len(self.stations[s]) >= self.min_robots_req or len(self.stations[s]) == 0:
                continue

            # if not enough available slots in other stations to move to, skip
            available_slots = sum(max_rps - len(st + self.stations_penalty[i])
                                  for i, st in enumerate(self.stations) if self.stations[s] != st and
                                  len(st + self.stations_penalty[i]) >= self.min_robots_req)
            if available_slots < len(self.stations[s] + self.stations_penalty[s]):
                continue

            robots_to_remove = []
            all_robots_in_s = self.stations[s] + self.stations_penalty[s]
            for robot in all_robots_in_s:
                # for each robot in the selected station, find the destination to move to that does not incur penalty
                current_stations_amount = len(self.stations)
                for i in range(current_stations_amount):
                    # skip if no available slot to move to, is its own station, or the dst is leq 5
                    if (len(self.stations[i] + self.stations_penalty[i]) >= max_rps or  # no available slot
                            self.stations[i] == self.stations[s] or  # dst is src
                            len(self.stations[i] + self.stations_penalty[i]) < self.min_robots_req):  # dst leq 5
                        continue

                    # if move robot to the dst station and incur no penalty, move it
                    tmp_dst_station = self.stations[i] + [robot]
                    tmp_centroid = self.find_weighted_centroid(tuple(tmp_dst_station))
                    if not self.contains_penalty(tmp_dst_station, tmp_centroid):
                        print(f"Move robot {robot} from station {s} to {i}")
                        self.stations[i].append(robot)
                        self.stations_loc[i] = tmp_centroid

                        robots_to_remove.append(robot)
                        break

            for r in robots_to_remove:
                if r in self.stations[s]:
                    self.stations[s].remove(r)
                else:
                    self.stations_penalty[s].remove(r)

            # if there are robots to be moved left
            if len(self.stations[s]) > 0:
                # try moving to station with odd number of robots first
                stations_odd_size = np.array([int(len(self.stations[i] + self.stations_penalty[i]) % self.q != 0 and
                                                  len(self.stations[i] + self.stations_penalty[i]) >=
                                                  self.min_robots_req) for i in range(len(self.stations))])
                insert_to_stations = np.argwhere(stations_odd_size != 0).flatten()
                if sum(stations_odd_size) >= len(self.stations[s]):
                    robots = self.stations[s].copy()
                    for r, robot in enumerate(robots):
                        self.stations_penalty[insert_to_stations[r]].append(robot)
                        self.stations[s].pop(0)
                elif insert_to_stations:
                    for inserted_station in insert_to_stations:
                        self.stations_penalty[inserted_station].append(self.stations[s].pop(0))

                # if no station with odd number of robots left --> only even number or full left
                current_robots_amount = len(self.stations[s])
                while current_robots_amount > 0:
                    # find stations that does not reach limit and has >5 robots
                    # resulting stations will have at most max_rps - 2 stations
                    stations_available = np.array([int(
                        max_rps > len(self.stations[i] + self.stations_penalty[i]) >= self.min_robots_req)
                        for i in range(len(self.stations))])

                    insert_to_stations = np.argwhere(stations_available != 0).flatten()
                    for i in insert_to_stations:
                        try:
                            # insert as a pair to reduce charger cost
                            self.stations_penalty[i].append(self.stations[s].pop(0))
                            self.stations_penalty[i].append(self.stations[s].pop(0))
                        except IndexError:
                            # if error --> no more robot left
                            break

                    current_robots_amount = len(self.stations[s])

            station_to_remove.append(s)

        for i, s in enumerate(station_to_remove):
            self.stations.pop(s-i)
            self.stations_penalty.pop(s-i)
            self.stations_loc.pop(s-i)

    def print_results(self):
        print(f"The improved cost is {self.find_total_cost(self.stations, self.stations_loc)}")
        print(f"Total stations: {len(self.stations)}")
        print(f"robot in stations: {self.stations}")
        print(f"penalty robot in stations: {self.stations_penalty}")
        print(f"stations location: {self.stations_loc}")

    def get_heuristics_results(self) -> HeuristicsResults:
        return HeuristicsResults(objective_value=self.find_total_cost(self.stations, self.stations_loc),
                                 stations_alloc=self.stations.copy(),
                                 stations_loc=self.stations_loc.copy(),
                                 stations_with_penalty=self.stations_penalty.copy())
