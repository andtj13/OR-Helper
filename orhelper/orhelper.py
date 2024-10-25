import pandas
from ortools.linear_solver import pywraplp
from ortools.graph.python import max_flow, min_cost_flow
import numpy as np
from typing import Literal
import warnings
from abc import ABC


class LPHelper(ABC):
    r"""
    Abstract base class for all variations of linear programming problems.  Child classes inherit basic functions but
over the add_constraints() method in order to accommodate differently structured DataFrames as inputs.
    """

    def __init__(self, direction: Literal['max', 'min'], constraint_df: pandas.DataFrame, objective_df: pandas.DataFrame):

        if direction not in {'max', 'min'}:
            raise ValueError(f"Invalid direction {direction}.  Direction must be either 'max' or 'min'.")
        self.constraint_df = constraint_df
        self.objective_df = objective_df
        self.solver = pywraplp.Solver.CreateSolver("GLOP")
        self.objective = self.solver.Objective()
        self.direction = direction

        self.variables = []
        self.constraints = {}
        self.dual_values = {}
        self.reduced_costs = {}
        self.optimal_value = None
        self.variable_optimal_value = {}

        self.run_lp()

    def __str__(self):
        output_string = ""
        if self.optimal_value is None:
            self.get_results()

        output_string = output_string + "Variable results:\n"

        for var in self.variable_optimal_value:
            output_string = output_string + f"{var} optimal value is {self.variable_optimal_value[var]}\n"

        output_string = output_string + f"\nOptimal value of the objective function is {self.optimal_value}\n"

        if len(self.dual_values) > 0:
            output_string = output_string + "\nShadow Prices/Dual Values:\n"
            for cons in self.dual_values:
                output_string = output_string + f"Shadow price of constraint {cons} = {self.dual_values[cons]}\n"

        if len(self.reduced_costs) > 0:
            output_string = output_string + "\nReduced Costs\n"
            for var in self.reduced_costs:
                output_string = output_string + f"Reduced cost of variable {var}  = {self.reduced_costs[var]}\n"

        return output_string

    def handle_infinity(self, input_list: list) -> list:
        r"""
        Helper function to convert various values into solver.Infinity() for or-tools.
        Excel sheets and DataFrames may contain various values that stand-in for infinity.
        This function expects the most obvious, but does not capture all.
        If not infinity or negative infinity, values are left as-is.
        :param input_list: List of values from a DataFrame to be converted, if infinity or negative infinity.
        :return: List of values with all infinity stand-ins converted to solver.Infinity().
        """
        inf_value_hashmap = {
            float('inf'): self.solver.Infinity(),
            np.Infinity: self.solver.Infinity(),
            'inf': self.solver.Infinity(),
            'infinity': self.solver.Infinity(),
            'Infinity': self.solver.Infinity(),
            float('-inf'): -self.solver.Infinity(),
            -np.Infinity: -self.solver.Infinity(),
            '-inf': -self.solver.Infinity(),
            '-infinity': -self.solver.Infinity(),
            '-Infinity': -self.solver.Infinity(),
            'neg inf': -self.solver.Infinity(),
            'neg infinity': -self.solver.Infinity(),
            'neg Infinity': -self.solver.Infinity(),
            'negative inf': -self.solver.Infinity(),
            'negative infinity': -self.solver.Infinity(),
            'Negative Infinity': -self.solver.Infinity(),
        }

        output_list = [inf_value_hashmap[x] if x in inf_value_hashmap else x for x in input_list]
        return output_list

    def set_objective_function(self):
        r"""
        Sets objective function, including variable coefficients for all linear programming problems.
        :return: Dictionary of variable names and coefficients.
        """

        variable_names = self.objective_df.iloc[:, 0].values.tolist()  # upack 1st column for variable names

        variable_lower_bounds = self.objective_df.iloc[:, 1].values.tolist()  # unpack 2nd column for lower bounds
        variable_lower_bounds = self.handle_infinity(variable_lower_bounds)

        variable_upper_bounds = self.objective_df.iloc[:, 2].values.tolist()  # unpack 3rd column for upper bounds
        variable_upper_bounds = self.handle_infinity(variable_upper_bounds)

        for i, name in enumerate(variable_names):
            self.variables.append(self.solver.NumVar(variable_lower_bounds[i], variable_upper_bounds[i], name))

        # need to set coefficients
        objective_coefficients = self.objective_df.iloc[:, 3].values.tolist()
        for i, vari in enumerate(self.variables):
            self.objective.SetCoefficient(vari, objective_coefficients[i])

        if self.direction == 'max':
            self.objective.SetMaximization()
        else:
            self.objective.SetMinimization()

        return self.variables, self.objective

    def get_dual_values(self) -> dict:
        r"""
        Positive dual values are binding, and increasing them will improve optimal solution.
        Zeros are non-binding and have no impact on result.
        Negative dual values will worsen the result.
        :return: Dictionary of dual values corresponding to each constraint.
        """
        for cons in self.constraints:
            self.dual_values[cons] = self.constraints[cons].dual_value()
        return self.dual_values

    def get_reduced_costs(self) -> dict:
        r"""
        The opportunity costs of each variable.
        If negative, increasing variable would decrease result.
        If positive, increasing variable would increase result.
        :return: Dict of reduced costs corresponding to each variable.
        """
        for var in self.variables:
            self.reduced_costs[var.name()] = var.reduced_cost()
        return self.reduced_costs

    def get_results(self, dual_values: bool = True, reduced_costs: bool = True):
        r"""
        Runs the ortool engine to solve objective function, including optimal values for each variable.
        :param dual_values:
        :param reduced_costs:
        :return:
        """
        status = self.solver.Solve()
        if status != self.solver.OPTIMAL:
            warnings.warn("No optimal solution found for objective function.")
            if status != self.solver.FEASIBLE:
                raise ValueError("Objective function does not have a feasible solution.")

        self.optimal_value = self.objective.Value()

        for var in self.variables:
            self.variable_optimal_value[var.name()] = var.solution_value()  # taking the name, not the entire variable

        if dual_values is True:
            self.get_dual_values()

        if reduced_costs is True:
            self.get_reduced_costs()

        return

    def add_constraints(self):
        return

    def run_lp(self):
        r"""
        Sets objective funciton, adds constraints, and gets results at initalization.
        :return:
        """
        self.set_objective_function()
        self.add_constraints()
        self.get_results()
        return


class LPHorizDfs(LPHelper):

    def add_constraints(self):
        r"""
        Adds constraints from a horizontally shaped dataframe.  The names of constraints
    are in the first column, and then lower bounds, upper bounds, and variables are
    named on the column headers.  See sample dataframe:

    >>>	costraint_names	    lower_bounds	upper_bounds	variable_1_coeff	variable_2_coeff
    >>> 0	constraint_1	        0	            150	                12	                4
    >>> 1	constraint_2	        0	            23	                18	                9
    >>> 2	constraint_3	        0	            98	                3	                7
    >>> 3	constraint_4	        0	            34	                2	                10
        """
        constraint_lower_bounds = self.constraint_df.iloc[:, 1].values.tolist()  # get lower bounds from 2nd column
        constraint_lower_bounds = self.handle_infinity(constraint_lower_bounds)

        constraint_upper_bounds = self.constraint_df.iloc[:, 2].values.tolist()  # get upper bounds from 3rd column
        constraint_upper_bounds = self.handle_infinity(constraint_upper_bounds)

        constraint_names = self.constraint_df.iloc[:, 0].to_list()

        for i, cons in enumerate(constraint_lower_bounds):
            cons_name = constraint_names[i]  # using the first column as the constraint names
            self.constraints[cons_name] = self.solver.Constraint(constraint_lower_bounds[i], constraint_upper_bounds[i])
            for j, vari in enumerate(self.variables):  # each variable needs a coefficient
                coefficient_of_variable = self.constraint_df.iloc[:, j + 3].values.tolist()[
                    i]  # get coefficient from next column
                self.constraints[cons_name].SetCoefficient(self.variables[j], coefficient_of_variable)

        return


class LPVertiDfs(LPHelper):

    def add_constraints(self):
        r"""
        Adds constraints from a vertically-structured dataframe.  Lower bounds, upper bounds,
        and variables populate the rows of the dataframe.  The first non-header row is lower bounds,
        second row is upper-bounds, and all other rows are variables.  Constraint names are found
        in column headers.  See sample dataframe.

        >>> 	Variable	    constraint_1	constraint_2	constraint_3	constraint_4
        >>> 0	lower_bounds	        0	            0	            0	            0
        >>> 1	upper_bounds	        150	            23	            98	            34
        >>> 2	variable_1_coeff	12	            18	            3	            2
        >>> 3	variable_2_coeff	4	            9	            10	            10
        :return:
        """
        constraint_df = self.constraint_df.drop(self.constraint_df.columns[0], axis=1)  # drop 1st column

        df_data = constraint_df.values.tolist()

        constraint_names = constraint_df.columns.tolist()  # get names for constraints from column headers

        constraint_lower_bounds = df_data[0]
        constraint_lower_bounds = self.handle_infinity(constraint_lower_bounds)

        constraint_upper_bounds = df_data[1]
        constraint_upper_bounds = self.handle_infinity(constraint_upper_bounds)

        for i, cons in enumerate(constraint_lower_bounds):
            cons_name = constraint_names[i] # taking column names for constraint names
            self.constraints[cons_name] = self.solver.Constraint(constraint_lower_bounds[i], constraint_upper_bounds[i])
            for j, vari in enumerate(self.variables):
                coefficient_of_variable = df_data[j+2][i]
                if not np.isnan(coefficient_of_variable):
                    self.constraints[cons_name].SetCoefficient(self.variables[j], coefficient_of_variable)
        return


class IPHelper(ABC):
    r"""
    Abstract base class for all variations of integer programming problems, using the CP-SAT solver. nChild classes
    inherit basic functions but over the add_constraints() method in order to accommodate differently structured
    DataFrames as inputs.
    """

    def __init__(self, direction: Literal['max', 'min'], constraint_df: pandas.DataFrame, objective_df: pandas.DataFrame):

        if direction not in {'max', 'min'}:
            raise ValueError(f"Invalid direction {direction}.  Direction must be either 'max' or 'min'.")
        self.constraint_df = constraint_df
        self.objective_df = objective_df
        self.solver = pywraplp.Solver.CreateSolver("SAT")
        self.objective = self.solver.Objective()
        self.direction = direction

        self.variables = []
        self.constraints = {}
        self.optimal_value = None
        self.variable_optimal_value = {}

        self.run_ip()

    def __str__(self):
        output_string = ""
        if self.optimal_value is None:
            self.get_results()

        output_string = output_string + "Variable results:\n"

        for var in self.variable_optimal_value:
            output_string = output_string + f"{var} optimal value is {self.variable_optimal_value[var]}\n"

        output_string = output_string + f"\nOptimal value of the objective function is {self.optimal_value}\n"

        return output_string

    def handle_infinity(self, input_list: list) -> list:
        r"""
        Helper function to convert various values into solver.Infinity() for or-tools.
        Excel sheets and DataFrames may contain various values that stand-in for infinity.
        This function expects the most obvious, but does not capture all.
        If not infinity or negative infinity, values are left as-is.
        :param input_list: List of values from a DataFrame to be converted, if infinity or negative infinity.
        :return: List of values with all infinity stand-ins converted to solver.Infinity().
        """
        inf_value_hashmap = {
            float('inf'): self.solver.Infinity(),
            np.Infinity: self.solver.Infinity(),
            'inf': self.solver.Infinity(),
            'infinity': self.solver.Infinity(),
            'Infinity': self.solver.Infinity(),
            float('-inf'): -self.solver.Infinity(),
            -np.Infinity: -self.solver.Infinity(),
            '-inf': -self.solver.Infinity(),
            '-infinity': -self.solver.Infinity(),
            '-Infinity': -self.solver.Infinity(),
            'neg inf': -self.solver.Infinity(),
            'neg infinity': -self.solver.Infinity(),
            'neg Infinity': -self.solver.Infinity(),
            'negative inf': -self.solver.Infinity(),
            'negative infinity': -self.solver.Infinity(),
            'Negative Infinity': -self.solver.Infinity(),
        }

        output_list = [inf_value_hashmap[x] if x in inf_value_hashmap else x for x in input_list]
        return output_list

    def set_objective_function(self):
        r"""
        :return:
        """

        variable_names = self.objective_df.iloc[:, 0].values.tolist()  # upack 1st column for variable names

        variable_lower_bounds = self.objective_df.iloc[:, 1].values.tolist()  # unpack 2nd column for lower bounds
        variable_lower_bounds = self.handle_infinity(variable_lower_bounds)

        variable_upper_bounds = self.objective_df.iloc[:, 2].values.tolist()  # unpack 3rd column for upper bounds
        variable_upper_bounds = self.handle_infinity(variable_upper_bounds)

        for i, name in enumerate(variable_names):
            self.variables.append(self.solver.IntVar(variable_lower_bounds[i], variable_upper_bounds[i], name))

        # need to set coefficients
        objective_coefficients = self.objective_df.iloc[:, 3].values.tolist()
        for i, vari in enumerate(self.variables):
            self.objective.SetCoefficient(vari, objective_coefficients[i])

        if self.direction == 'max':
            self.objective.SetMaximization()
        else:
            self.objective.SetMinimization()

        return self.variables, self.objective

    def get_results(self):
        r"""
        Runs the ortool engine to solve objective function, including optimal values for each variable.

        :param dual_values:
        :param reduced_costs:
        :return:
        """
        status = self.solver.Solve()
        if status != self.solver.OPTIMAL:
            warnings.warn("No optimal solution found for objective function.")
            if status != self.solver.FEASIBLE:
                raise ValueError("Objective function does not have a feasible solution.")

        self.optimal_value = self.objective.Value()

        for var in self.variables:
            self.variable_optimal_value[var.name()] = var.solution_value()  # taking the name, not the entire variable

        return

    def add_constraints(self):
        return

    def run_ip(self):
        self.set_objective_function()
        self.add_constraints()
        self.get_results()
        return


class IPHorizDfs(IPHelper):

    def add_constraints(self):
        r"""
        Adds constraints from a horizontally shaped dataframe.  The names of constraints
    are in the first column, and then lower bounds, upper bounds, and variables are
    named on the column headers.  See sample dataframe:

    >>>	costraint_names	    lower_bounds	upper_bounds	variable_1_coeff	variable_2_coeff
    >>> 0	constraint_1	        0	            150	                12	                4
    >>> 1	constraint_2	        0	            23	                18	                9
    >>> 2	constraint_3	        0	            98	                3	                7
    >>> 3	constraint_4	        0	            34	                2	                10
        """
        constraint_lower_bounds = self.constraint_df.iloc[:, 1].values.tolist()  # get lower bounds from 2nd column
        constraint_lower_bounds = self.handle_infinity(constraint_lower_bounds)

        constraint_upper_bounds = self.constraint_df.iloc[:, 2].values.tolist()  # get upper bounds from 3rd column
        constraint_upper_bounds = self.handle_infinity(constraint_upper_bounds)

        constraint_names = self.constraint_df.iloc[:, 0].to_list()

        for i, cons in enumerate(constraint_lower_bounds):
            cons_name = constraint_names[i]  # using the first column as the constraint names
            self.constraints[cons_name] = self.solver.Constraint(constraint_lower_bounds[i], constraint_upper_bounds[i])
            for j, vari in enumerate(self.variables):  # each variable needs a coefficient
                coefficient_of_variable = self.constraint_df.iloc[:, j + 3].values.tolist()[
                    i]  # get coefficient from next column
                self.constraints[cons_name].SetCoefficient(self.variables[j], coefficient_of_variable)

        return


class IPVertiDfs(IPHelper):

    def add_constraints(self):
        r"""
        Adds constraints from a vertically-structured dataframe.  Lower bounds, upper bounds,
        and variables populate the rows of the dataframe.  The first non-header row is lower bounds,
        second row is upper-bounds, and all other rows are variables.  Constraint names are found
        in column headers.  See sample dataframe.

        >>> 	Variable	    constraint_1	constraint_2	constraint_3	constraint_4
        >>> 0	lower_bounds	        0	            0	            0	            0
        >>> 1	upper_bounds	        150	            23	            98	            34
        >>> 2	variable_1_coeff	12	            18	            3	            2
        >>> 3	variable_2_coeff	4	            9	            10	            10
        :return:
        """
        constraint_df = self.constraint_df.drop(self.constraint_df.columns[0], axis=1)  # drop 1st column

        df_data = constraint_df.values.tolist()

        constraint_names = constraint_df.columns.tolist()  # get names for constraints from column headers

        constraint_lower_bounds = df_data[0]
        constraint_lower_bounds = self.handle_infinity(constraint_lower_bounds)

        constraint_upper_bounds = df_data[1]
        constraint_upper_bounds = self.handle_infinity(constraint_upper_bounds)

        for i, cons in enumerate(constraint_lower_bounds):
            cons_name = constraint_names[i] # taking column names for constraint names
            self.constraints[cons_name] = self.solver.Constraint(constraint_lower_bounds[i], constraint_upper_bounds[i])
            for j, vari in enumerate(self.variables):
                coefficient_of_variable = df_data[j+2][i]
                if not np.isnan(coefficient_of_variable):
                    self.constraints[cons_name].SetCoefficient(self.variables[j], coefficient_of_variable)
        return


class MaxFlow:

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.capacities = self.dataframe.iloc[:, 2].to_numpy()
        self.solver = max_flow.SimpleMaxFlow()
        self.all_arcs = self.parse_dataframe_and_solve()  # assigned later
        self.optimal_flow = self.solver.optimal_flow() # assigned later
        self.source_side_min_cut = self.solver.get_source_side_min_cut()
        self.sink_side_min_cut = self.solver.get_sink_side_min_cut()
        self.solution_flows = self.solver.flows(self.all_arcs)

    def __str__(self):
        output_string = f"Max Flow: {self.optimal_flow}.\n"

        for arc, flow, capacity in zip(self.all_arcs, self.solution_flows, self.capacities):
            output_string = output_string + f"{self.solver.tail(arc)} -> {self.solver.head(arc)}    {flow} / {capacity}.\n"

        output_string = output_string + f"Source side min-cut: {self.source_side_min_cut}\n"
        output_string = output_string + f"Sink side min-cut: {self.sink_side_min_cut}\n"

        return output_string

    def parse_dataframe_and_solve(self):
        start_nodes = self.dataframe.iloc[:, 0].to_numpy()
        end_nodes = self.dataframe.iloc[:, 1].to_numpy()
        capacities = self.dataframe.iloc[:, 2].to_numpy()

        source = start_nodes[0]
        sink = end_nodes[-1]

        all_arcs = self.solver.add_arcs_with_capacity(start_nodes, end_nodes, capacities)

        status = self.solver.solve(source, sink)

        if status != self.solver.OPTIMAL:
            warnings.warn(f"Non-optimal solution found - status {status}.")

        return all_arcs


class MinFlow:

    def __init__(self, graph_dataframe, supplies_df):
        self.graph_dataframe = graph_dataframe
        self.supplies = supplies_df.iloc[:, 0].to_numpy()
        self.start_nodes = self.graph_dataframe.iloc[:, 0].to_numpy()
        self.end_nodes = self.graph_dataframe.iloc[:, 1].to_numpy()
        self.capacities = self.graph_dataframe.iloc[:, 2].to_numpy()
        self.unit_costs = self.graph_dataframe.iloc[:, 3].to_numpy()
        self.solver = min_cost_flow.SimpleMinCostFlow()
        self.all_arcs = self.parse_dataframe_and_solve()  # assigned later
        self.optimal_cost = self.solver.optimal_cost()
        self.solution_flows = self.solver.flows(self.all_arcs)

    def __str__(self):
        output_string = f"Max Flow: {self.optimal_cost}.\n"

        for arc, flow, capacity, cost in zip(self.all_arcs, self.solution_flows, self.capacities, self.unit_costs):
            output_string = output_string + (f"{self.solver.tail(arc)} -> {self.solver.head(arc)}    {flow} /"
                                             f" {capacity}.     Cost: {cost}\n")

        return output_string

    def parse_dataframe_and_solve(self):

        all_arcs = self.solver.add_arcs_with_capacity_and_unit_cost(
            self.start_nodes, self.end_nodes, self.capacities, self.unit_costs
        )

        self.solver.set_nodes_supplies(
            np.arange(0, len(self.supplies)), self.supplies
        )

        status = self.solver.solve()

        if status != self.solver.OPTIMAL:
            warnings.warn(f"Non-optimal solution found - status {status}.")

        return all_arcs
