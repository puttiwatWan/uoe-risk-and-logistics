import numpy as np

from atcs import ATCS
from utils import set_print_time, time_spent_decorator
from config import config

from heuristics.heuristics import ConstructionHeuristicSolver, ImprovementCentroidHeuristics


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
    solver = ConstructionHeuristicSolver(robot_range=robot_range,
                                         robot_loc=robot_loc,
                                         robot_distance_matrix=dist_matrix)

    solver.solve(starting_robot=starting_robot)
    solver.print_results()

    results = solver.get_heuristics_results()

    print("_" * 60)
    imp_centroid_solver = ImprovementCentroidHeuristics(robot_range=robot_range,
                                                        robot_loc=robot_loc,
                                                        robot_distance_matrix=dist_matrix,
                                                        results=results)

    imp_centroid_solver.solve()
    imp_centroid_solver.print_results()


if __name__ == "__main__":
    main()
