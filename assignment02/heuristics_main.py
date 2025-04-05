import numpy as np

from atcs import ATCS
from utils import print_separator, set_print_time, time_spent_decorator
from config import config

from heuristics.heuristics import ConstructionHeuristicSolver, ImprovementCentroidHeuristics, \
    ImprovementStationsReductionHeuristics


@time_spent_decorator
def main():
    # Initialize Data
    print_time = config.print_time
    seed = config.seed
    use_subset_robot = config.use_subset_robot
    n_samples = config.n_samples
    starting_robot = config.default_starting_robot

    set_print_time(print_time)
    data = ATCS(seed=seed)

    robot_loc = data.l_df.to_numpy()
    robot_range = data.r_df.to_numpy().flatten()
    dist_matrix = data.get_distance_matrix(sample_subset=False)
    if use_subset_robot:
        data.choose_subset_point(n_samples, randomized=False)  # Choose subset data
        robot_loc = data.l_sub_df.to_numpy()
        robot_range = data.r_sub_df.to_numpy().flatten()
        dist_matrix = data.get_distance_matrix(sample_subset=True)

    robot_loc = np.array(robot_loc)
    robot_range = np.array(robot_range)

    # Solve Construction Heuristics
    print_separator("Construction Heuristics")
    solver = ConstructionHeuristicSolver(robot_range=robot_range,
                                         robot_loc=robot_loc,
                                         robot_distance_matrix=dist_matrix)

    solver.solve(starting_robot=starting_robot)
    solver.print_results()

    results = solver.get_heuristics_results()

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



if __name__ == "__main__":
    main()
