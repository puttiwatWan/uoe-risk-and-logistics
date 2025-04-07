from atcs import ATCS
from heuristics.stochastic import StochasticConstructionHeuristicSolver, StochasticImprovementCentroidHeuristics, \
    StochasticImprovementStationsReductionHeuristics, StochasticImprovementLocalSearchHeuristics
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
    dist_matrix = data.distance_matrix.copy()
    robot_scenario_used = data.cc_df.to_numpy()
    expected_range = data.expected_range.to_numpy().flatten()
    if use_subset_robot:
        data.choose_subset_point(n_samples, randomized=False)  # Choose subset data
        robot_loc = data.l_sub_df.to_numpy()
        robot_range_scenarios = data.r_s_sub_df.to_numpy()
        robot_scenario_used = data.cc_sub_df.to_numpy()
        expected_range = data.expected_range_sub.to_numpy().flatten()
        dist_matrix = data.distance_matrix_sub.copy()

    # Solve Construction Heuristics
    print_separator("Construction Heuristics")
    solver = StochasticConstructionHeuristicSolver(robot_loc=robot_loc,
                                                   robot_expected_range=expected_range,
                                                   robot_distance_matrix=dist_matrix,
                                                   robot_range_scenarios=robot_range_scenarios,
                                                   robot_scenario_used=robot_scenario_used,)

    solver.solve(starting_robot=starting_robot)
    solver.print_results()

    results = solver.get_heuristics_results()

    print_separator("Improving Centroid Heuristics")
    solver = StochasticImprovementCentroidHeuristics(robot_expected_range=expected_range,
                                                     robot_loc=robot_loc,
                                                     robot_distance_matrix=dist_matrix,
                                                     robot_range_scenarios=robot_range_scenarios,
                                                     robot_scenario_used=robot_scenario_used,
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
        solver = StochasticImprovementStationsReductionHeuristics(robot_expected_range=expected_range,
                                                                  robot_loc=robot_loc,
                                                                  robot_distance_matrix=dist_matrix,
                                                                  robot_range_scenarios=robot_range_scenarios,
                                                                  robot_scenario_used=robot_scenario_used,
                                                                  results=results)

        solver.solve()
        solver.print_results()

        iter += 1
        prev_value = results.objective_value
        results = solver.get_heuristics_results()
        contain_leq_five = min([len(station) for station in results.stations]) <= 5

    print_separator("Improving Local Search Heuristics")
    solver = StochasticImprovementLocalSearchHeuristics(robot_expected_range=expected_range,
                                                        robot_loc=robot_loc,
                                                        robot_distance_matrix=dist_matrix,
                                                        robot_range_scenarios=robot_range_scenarios,
                                                        robot_scenario_used=robot_scenario_used,
                                                        results=results)

    solver.solve()
    solver.print_results()


if __name__ == "__main__":
    main()
