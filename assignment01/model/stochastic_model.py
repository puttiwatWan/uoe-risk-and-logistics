import pandas as pd
import xpress as xp

from utils import time_spent_decorator


class StochasticModel:
    def __init__(self,
                 agg_dem_cus_period_clus_scene_df: pd.DataFrame,
                 candidate_df: pd.DataFrame,
                 supplier_df: pd.DataFrame,
                 cluster_center_df: pd.DataFrame,
                 cost_w_to_s: pd.DataFrame,
                 cost_w_to_cluster: pd.DataFrame):
        self.agg_dem_cus_period_clus_scene_df = agg_dem_cus_period_clus_scene_df.copy()
        self.candidate_df = candidate_df.copy()
        self.supplier_df = supplier_df.copy()
        self.cluster_center_df = cluster_center_df.copy()
        self.cost_w_to_s = cost_w_to_s.copy()
        self.cost_w_to_cluster = cost_w_to_cluster.copy()
        self.time_limit_in_sec = 3600

        self.S = list(supplier_df.index)  # Supplier Index
        S_P0 = list(supplier_df[supplier_df['SupplierProductGroup'] == 0]['SupplierProductGroup'].index)
        S_P1 = list(supplier_df[supplier_df['SupplierProductGroup'] == 1]['SupplierProductGroup'].index)
        S_P2 = list(supplier_df[supplier_df['SupplierProductGroup'] == 2]['SupplierProductGroup'].index)
        S_P3 = list(supplier_df[supplier_df['SupplierProductGroup'] == 3]['SupplierProductGroup'].index)
        self.S_P_dict = {0: S_P0, 1: S_P1, 2: S_P2, 3: S_P3}

        self.equal_prob = 1 / len(self.agg_dem_cus_period_clus_scene_df.columns)

        self.W = list(candidate_df.index)  # Warehouse Index
        self.C = list(cluster_center_df.index)  # Cluster Index
        self.P = list(agg_dem_cus_period_clus_scene_df.reset_index()['ProductIndex'].unique())
        self.T = list(agg_dem_cus_period_clus_scene_df.reset_index()['PeriodIndex'].unique())
        self.Phi = list(agg_dem_cus_period_clus_scene_df.columns)

        self.model = xp.problem(name='MEWLP_Deterministic')

    @time_spent_decorator
    def __init_variables(self):
        self.x = self.model.addVariables(self.W, self.C, self.T, self.Phi, name="x", vartype=xp.binary)
        self.y = self.model.addVariables(self.W, name="y", vartype=xp.binary)
        self.o = self.model.addVariables(self.W, self.T, name="o", vartype=xp.binary)
        self.z = self.model.addVariables(self.W, self.S, self.T, name="z", vartype=xp.continuous)

    @time_spent_decorator
    def __generate_constraints(self):
        self.model.addConstraint(self.y[w] >= self.o[w, t] for w in self.W for t in self.T)
        self.model.addConstraint(self.o[w, t] >= self.o[w, t - 1] for w in self.W for t in self.T if t != 0)
        self.model.addConstraint(
            self.x[w, c, t, xi] <= self.o[w, t] for w in self.W for t in self.T for c in self.C for xi in self.Phi)

        # Limit warehouse capacity >= total covered demand
        self.model.addConstraint(xp.Sum(self.agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * self.x[w, c, t, xi]
                                        for c in self.C for p in self.P)
                                 <= self.candidate_df.loc[w, 'Capacity'] * self.o[w, t]
                                 for w in self.W for t in self.T for xi in self.Phi)

        # A customer needs to be covered by a warehouse
        self.model.addConstraint(
            xp.Sum(self.x[w, c, t, xi] for w in self.W) == 1 for c in self.C for t in self.T for xi in self.Phi)

        self.model.addConstraint(xp.Sum(self.z[w, s, t] for s in self.S_P_dict[p]) >=
                                 xp.Sum(
                                     self.agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * self.x[w, c, t, xi] for
                                     c in self.C)
                                 for w in self.W for p in self.P for t in self.T for xi in self.Phi)

        self.model.addConstraint(
            xp.Sum(self.z[w, s, t] for w in self.W) <= self.supplier_df.loc[s, 'SupplierCapacity'] for s in self.S for t
            in self.T)

    @time_spent_decorator
    def __set_objective(self):
        Setup_cost = xp.Sum(self.candidate_df.loc[w, 'Setup'] * self.y[w] for w in self.W)
        Operating_cost = xp.Sum(self.candidate_df.loc[w, 'Operating'] * self.o[w, t] for w in self.W for t in self.T)
        Tra_w_s_cost = xp.Sum(self.cost_w_to_s.loc[w, s] * self.z[w, s, t]
                              for w in self.W for s in self.S for t in self.T)

        Recourse = xp.Sum(self.equal_prob * xp.Sum(self.cost_w_to_cluster.loc[w, c] *
                                                   self.agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] *
                                                   self.x[w, c, t, xi]
                                                   for w in self.W for c in self.C for p in self.P for t in self.T)
                          for xi in self.Phi)

        obj = Setup_cost + Operating_cost + Tra_w_s_cost + Recourse
        self.model.setObjective(obj, sense=xp.minimize)

    @time_spent_decorator
    def __solve(self):
        self.model.setControl('miprelstop', 1e-3)
        self.model.setControl('maxtime', self.time_limit_in_sec)

        # Solve the problem
        self.model.solve()

    @time_spent_decorator
    def __print_results(self):
        objective_value = self.model.getObjVal()
        mip_gap_percentage = 100 * (objective_value - self.model.getAttrib('bestbound')) / objective_value

        x_sol = self.model.getSolution(self.x)
        y_sol = self.model.getSolution(self.y)
        o_sol = self.model.getSolution(self.o)
        z_sol = self.model.getSolution(self.z)

        print(f'Objective Value: {objective_value}')
        print(f'%Gaps: {mip_gap_percentage}')

        set_cost_sol = sum(self.candidate_df.loc[w, 'Setup'] * y_sol[w] for w in self.W)
        operation_cost_sol = sum(self.candidate_df.loc[w, 'Operating'] * o_sol[w, t] for w in self.W for t in self.T)
        w_s_cost_sol = sum(self.cost_w_to_s.loc[w, s] * z_sol[w, s, t] for w in self.W for s in self.S for t in self.T)
        recourse_sol = sum(self.equal_prob * sum(self.cost_w_to_cluster.loc[w, c] *
                                                 self.agg_dem_cus_period_clus_scene_df.loc[(c, p, t), xi] * x_sol[
                                                     w, c, t, xi]
                                                 for w in self.W for c in self.C for p in self.P for t in self.T)
                           for xi in self.Phi)

        print(f"Setup cost: {set_cost_sol}")
        print(f"Operation cost: {operation_cost_sol}")
        print(f"Transportation from Supplier to Warehouse cost: {w_s_cost_sol}")
        print(f"Recourse cost: {recourse_sol}")

        opened_warehouse = []
        for t in self.T:
            setup_warehouse_in_t = [w for w in self.W if
                                    y_sol[w] == 1 and o_sol[w, t] == 1 and w not in opened_warehouse]
            print(f"Warehouse setup in T: {t} are: {setup_warehouse_in_t}")
            opened_warehouse += setup_warehouse_in_t

    def run(self, time_limit_s: int = 3600):
        self.time_limit_in_sec = time_limit_s
        xp.init('/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr')

        self.__init_variables()
        self.__generate_constraints()
        self.__set_objective()
        self.__solve()
        self.__print_results()
