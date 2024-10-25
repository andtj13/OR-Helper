from orhelper import orhelper
import pandas as pd


def create_objective_function_df() -> pd.DataFrame:
    r"""
    Creates a dataframe of the objective function, using the problem in Williams 4.1 as an example.
    :return:
    """
    obj_func_dict = {
        "variable_names": ["FactA_Standard", "FactA_Deluxe", "FactB_Standard", "FactB_Deluxe"],  # name of each product
        "lower_bounds": [0, 0, 0, 0],
        "upper_bounds": ["inf", "inf", "inf", "inf"],
        "coefficient": [10, 15, 10, 15]  # profit value of each product
    }
    obj_func_df = pd.DataFrame.from_dict(obj_func_dict)
    return obj_func_df


def create_vertical_constraints_df() -> pd.DataFrame:
    r"""
    Create a vertical dataframe of constraints with lower bounds, upper bounds, and coefficients.  Using problem in
    Williams 4.1 as an example.
    :return:
    """
    vert_cnst_dct = {
        "variables": ["lower_bounds", "upper_bounds", "FactA_Standard", "FactA_Deluxe", "FactB_Standard", "FactB_Deluxe"],
        "GrindingFactA": [0, 80, 4, 2, 0, 0],  # grinding capacity and labor cost for Factory A
        "PolishingFactA": [0, 61, 2, 5, 0, 0],  # polishing capacity and labor cost for Factory A
        "GrindingFactB": [0, 60, 0, 0, 5, 5],  # grinding capacity and labor cost for Factory B
        "PolishingFactB": [0, 75, 0, 0, 3, 6],  # polishing capacity and labor cost for Factory A
        "RawMaterial": [0, 120, 4, 4, 4, 4]  # raw material
    }

    vert_cnst_df = pd.DataFrame.from_dict(vert_cnst_dct)
    return vert_cnst_df


def create_horizontal_constraints_df():

    horiz_cnst_dct = {
        "Constraint Names": ["GrindingFactA", "PolishingFactA", "GrindingFactB", "PolishingFactB", "RawMaterial"],
        "lower_bounds": [0, 0, 0, 0, 0],
        "upper_bounds": [80, 61, 60, 75, 120],
        "FactA_Standard": [4, 2, 0, 0, 4],
        "FactA_Deluxe": [2, 5, 0, 0, 4],
        "FactB_Standard": [0, 0, 5, 3, 4],
        "FactB_Deluxe": [0, 0, 5, 6, 4],
    }

    horiz_cnst_df = pd.DataFrame.from_dict(horiz_cnst_dct)

    return horiz_cnst_df


def check_optimal_value_from_williams41(opt_val: float):
    assert opt_val == 401.66666666666663
    return


def check_variable_optimal_values_wiliams41(var_opt_val: dict):
    assert var_opt_val["FactA_Standard"] == 9.66666666666667
    assert var_opt_val["FactA_Deluxe"] == 8.33333333333333
    assert var_opt_val["FactB_Standard"] == 0.0
    assert var_opt_val["FactB_Deluxe"] == 12.0
    return


def check_dual_values_williams41(dual_vals):
    assert dual_vals["GrindingFactA"] == 0.0
    assert dual_vals["PolishingFactA"] == 1.6666666666666663
    assert dual_vals["GrindingFactB"] == 1.6666666666666667
    assert dual_vals["PolishingFactB"] == 0.0
    assert dual_vals["RawMaterial"] == 1.6666666666666672
    return


def check_reduced_costs(red_costs):
    assert red_costs["FactA_Standard"] == -1.7763568394002505e-15
    assert red_costs["FactA_Deluxe"] == 0.0
    assert red_costs["FactB_Standard"] == -5.0000000000000036
    assert red_costs["FactB_Deluxe"] == -3.552713678800501e-15
    return


def test_vertical_df_lp():
    objective_func_df = create_objective_function_df()
    vertical_constraint_df = create_vertical_constraints_df()

    orh = orhelper.LPVertiDfs(direction='max', constraint_df=vertical_constraint_df, objective_df=objective_func_df)

    check_optimal_value_from_williams41(orh.optimal_value)
    check_variable_optimal_values_wiliams41(orh.variable_optimal_value)
    check_dual_values_williams41(orh.dual_values)
    check_reduced_costs(orh.reduced_costs)

    return


def test_horizontal_df_lp():
    objective_func_df = create_objective_function_df()
    horizontal_df = create_horizontal_constraints_df()

    orh = orhelper.LPHorizDfs(direction='max', constraint_df=horizontal_df, objective_df=objective_func_df)

    check_optimal_value_from_williams41(orh.optimal_value)
    check_variable_optimal_values_wiliams41(orh.variable_optimal_value)
    check_dual_values_williams41(orh.dual_values)
    check_reduced_costs(orh.reduced_costs)

    return


# Max Flow Testing
def create_sample_df_max_flow():
    sample_dict = {
        'start_nodes': [0, 0, 0, 1, 1, 2, 2, 3, 3],
        'end_nodes': [1, 2, 3, 2, 4, 3, 4, 2, 4],
        'capacities': [20, 30, 10, 40, 30, 10, 20, 5, 20],
    }

    df = pd.DataFrame.from_dict(sample_dict)

    return df


def test_max_flow():
    df = create_sample_df_max_flow()
    x = orhelper.MaxFlow(df)
    assert x.optimal_flow == 60
    assert x.source_side_min_cut[0] == 0
    assert x.sink_side_min_cut[0] == 4
    assert x.sink_side_min_cut[1] == 1


def create_min_cost_dataframes():
    min_flow_dict = {
        'start_nodes': [0, 0, 1, 1, 1, 2, 2, 3, 4],
        'end_nodes': [1, 2, 2, 3, 4, 3, 4, 4, 2],
        'capacities': [15, 8, 20, 4, 10, 15, 4, 20, 5],
        'unit_costs': [4, 4, 2, 2, 6, 1, 3, 2, 3]
    }

    supplies_dict = {
        'supplies': [20, 0, 0, -5, -15]
    }

    graph_df = pd.DataFrame.from_dict(min_flow_dict)
    supplies_df = pd.DataFrame.from_dict(supplies_dict)
    return graph_df, supplies_df


def test_min_flow():
    mf_df, supp_df = create_min_cost_dataframes()
    orh = orhelper.MinFlow(graph_dataframe=mf_df, supplies_df=supp_df)
    assert orh.optimal_cost == 150
    return


def create_ip_obj_func_dataframe():
    obj_func_dict = {
        "variable_names": ["x", "y"],
        "lower_bounds": [0, 0],
        "upper_bounds": ['inf', 'inf'],
        "coefficients": [1, 10],
    }

    obj_func_df = pd.DataFrame.from_dict(obj_func_dict)
    return obj_func_df


def create_vert_ip_constraints():
    cons_dict = {
        "variables": ["lower_bounds", "upper_bounds", "x", "y"],
        "first_constraint": ['-inf', 17.5, 1, 7],
        "second_constraint": ['-inf', 3.5, 1, 0]

    }

    cons_vert_df = pd.DataFrame.from_dict(cons_dict)
    return cons_vert_df


def test_vert_ip():
    ob_func = create_ip_obj_func_dataframe()
    constraints = create_vert_ip_constraints()
    orh = orhelper.IPVertiDfs(direction='max', constraint_df=constraints, objective_df=ob_func)
    assert orh.optimal_value == 23
    assert orh.variable_optimal_value['x'] == 3
    assert orh.variable_optimal_value['y'] == 2


def create_horiz_ip_constraints():
    cons_dict = {
        "Constraint Names": ["first_cons", "second_cons"],
        "lower_bounds": ["-inf", "-inf"],
        "upper_bounds": [17.5, 3.5],
        "x": [1, 1],
        "y": [7, 0],

    }

    cons_horiz_df = pd.DataFrame.from_dict(cons_dict)
    return cons_horiz_df


def test_horiz_ip():
    ob_func = create_ip_obj_func_dataframe()
    constraints = create_horiz_ip_constraints()
    orh = orhelper.IPHorizDfs(direction='max', constraint_df=constraints, objective_df=ob_func)
    assert orh.optimal_value == 23
    assert orh.variable_optimal_value['x'] == 3
    assert orh.variable_optimal_value['y'] == 2


if __name__ == '__main__':
    test_vertical_df_lp()
    test_horizontal_df_lp()
    test_max_flow()
    test_vert_ip()
    test_horiz_ip()
    x = create_sample_df_max_flow()
    print(x)