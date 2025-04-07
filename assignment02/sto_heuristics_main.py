from atcs import ATCS
from heuristics.stochastic import StochasticConstructionHeuristicSolver
from utils import print_separator, set_print_time, time_spent_decorator
from config import config


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
    robot_range_scenarios = data.r_s_df.to_numpy()
    dist_matrix = data.get_distance_matrix(sample_subset=False)
    robot_scenario_used = data.cc_df.to_numpy()
    expected_range = data.expected_range.to_numpy()
    if use_subset_robot:
        data.choose_subset_point(n_samples, randomized=False)  # Choose subset data
        robot_loc = data.l_sub_df.to_numpy()
        robot_range_scenarios = data.r_s_sub_df.to_numpy()
        robot_scenario_used = data.cc_sub_df.to_numpy()
        expected_range = data.expected_range_sub.to_numpy()
        dist_matrix = data.get_distance_matrix(sample_subset=use_subset_robot)

    print(robot_scenario_used.shape)

    # Solve Construction Heuristics
    print_separator("Construction Heuristics")
    solver = StochasticConstructionHeuristicSolver(robot_range_scenarios=robot_range_scenarios,
                                                   robot_loc=robot_loc,
                                                   robot_expected_range=expected_range,
                                                   robot_distance_matrix=dist_matrix,
                                                   robot_scenario_used=robot_scenario_used,)

    solver.solve(starting_robot=starting_robot)
    solver.print_results()

    results = solver.get_heuristics_results()


if __name__ == "__main__":
    main()
