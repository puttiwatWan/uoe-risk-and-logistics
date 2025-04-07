import copy

import numpy as np

from atcs import ATCS
from utils import print_separator, set_print_time, time_spent_decorator
from config import config

from heuristics.deterministic import ConstructionHeuristicSolver, ImprovementCentroidHeuristics, \
    ImprovementStationsReductionHeuristics, ImprovementLocalSearchHeuristics


@time_spent_decorator
def main():
    # Initialize Data
    print_time = config.print_time
    seed = config.seed
    use_subset_robot = config.use_subset_robot
    n_samples = config.n_samples

    set_print_time(print_time)
    data = ATCS(seed=seed)

    robot_loc = data.l_df.to_numpy()
    robot_range = data.r_df.to_numpy().flatten()
    dist_matrix = data.distance_matrix.copy()
    if use_subset_robot:
        data.choose_subset_point(n_samples, randomized=False)  # Choose subset data
        robot_loc = data.l_sub_df.to_numpy()
        robot_range = data.r_sub_df.to_numpy().flatten()
        dist_matrix = data.distance_matrix_sub.copy()

    robot_loc = np.array(robot_loc)
    robot_range = np.array(robot_range)

    # random_start = data.random_start
    random_start = [config.default_starting_robot]
    results = None
    best_solver = None
    for r in random_start:
        # Solve Construction Heuristics
        print_separator(f"Construction Heuristics Starting at Robot {r}")
        solver = ConstructionHeuristicSolver(robot_range=robot_range,
                                             robot_loc=robot_loc,
                                             robot_distance_matrix=dist_matrix)

        solver.solve(starting_robot=r)
        solver.print_results()
        tmp_results = solver.get_heuristics_results()
        if results is None or results.objective_value > tmp_results.objective_value:
            best_solver = copy.deepcopy(solver)
            results = copy.deepcopy(tmp_results)

    print("Best results is: ")
    best_solver.print_results()

    print_separator("Improving Centroid Heuristics")
    solver = ImprovementCentroidHeuristics(robot_range=robot_range,
                                           robot_loc=robot_loc,
                                           robot_distance_matrix=dist_matrix,
                                           results=results)

    solver.solve()
    solver.print_results()
    results = solver.get_heuristics_results()
    contain_leq_five = min([len(station) for station in results.stations]) <= 5

    # Improving Stations Reduction Heuristics
    iter = 1
    prev_value = 1
    while contain_leq_five and results.objective_value != prev_value:
        print_separator(f"Improving Stations Reduction Heuristics Iter {iter}")
        solver = ImprovementStationsReductionHeuristics(robot_range=robot_range,
                                                        robot_loc=robot_loc,
                                                        robot_distance_matrix=dist_matrix,
                                                        results=results)

        solver.solve()
        solver.print_results()

        iter += 1
        prev_value = results.objective_value
        results = solver.get_heuristics_results()
        contain_leq_five = min([len(station) for station in results.stations]) <= 5

    print_separator("Improving Local Search Heuristics")
    solver = ImprovementLocalSearchHeuristics(robot_range=robot_range,
                                              robot_loc=robot_loc,
                                              robot_distance_matrix=dist_matrix,
                                              results=results)

    solver.solve()
    solver.print_results()


if __name__ == "__main__":
    main()
