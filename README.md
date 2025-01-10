# OR Helper

Helper functions to feed pandas DataFrames into Google OR-Tools for linear programming, integer programm, and min/max flow problems.  

The purpose is to improve interoperability between pandas DataFrames or Excel spreadsheets and OR-Tool, along with additional features
such as constraint naming and handling infinity conversions.

## Linear Programming and Integer Programming

Linear programming and integer programming models are available to run optimization models directly from pandas 
DataFrames.  The solver is programmed to solve the problem as soon as the class is initialized.  

Constraint DataFrames may be input in two different forms: as horizontal DataFrames or as vertical DataFrames  This is a
convention meant to be maximally permissive with user input.  In cases where there are more constraints than variables,
it may make sense to use a horizontal DataFrame.  When the inverse is true, a vertical DataFrame is likely preferable.

### Horizontal Constraints
A horizontal DataFrame uses the first column for constraint names, the second column for constraint lower bounds, 
the third column for constraint upper bounds, and all remaining columns for the variable coefficients that correspond
to each constraint.  It is termed horizontal because as the model adds variables, the dataframe will grow wider.
In general, data would be structured this way when the model has more constraints than variables.

### Sample Horizontal Dataframe

```python
>>>	costraint_names	    lower_bounds	upper_bounds	variable_1_coeff	variable_2_coeff    
>>> 0	constraint_1	        0	            150	                12	                4           
>>> 1	constraint_2	        0	            23	                18	                9           
>>> 2	constraint_3	        0	            98	                3	                7           
>>> 3	constraint_4	        0	            34	                2	                10        
```

### Loading a Horizontal Dataframe using orhelper

```python
from orhelper import orhelper
import pandas as pd

# In this example, variables and constraints are split into different worksheets.
horiz_constraints_df = pd.read_excel("test.xlsx", sheet_name="constraints")
hoirz_var_df = pd.read_excel("test.xlsx", sheet_name="variables")

orh = orhelper.LPHorizDfs(
    direction='min', 
    constraint_df=horiz_constraints_df, 
    objective_df=hoirz_var_df
)

print(orh) # the __str__ method will print all relevant information about the solution
```

### Vertical Constraints
A vertical DataFrame organizes data in rows instead of columns.  The column headers should use the names of the 
constraints, if available, except for the first column header which can be left with a placeholder.  The first row 
should contain the lower bounds of each constraint, the second row the upper bounds, and all succeeding rows should 
contain the coefficients corresponding to each variable (e.g. variable 1 takes the third row, variable 2 takes the 
fourth, and so on).

```python
>>> 	Variable	    constraint_1	constraint_2	constraint_3	constraint_4
>>> 0	lower_bounds	        0	            0	            0	            0
>>> 1	upper_bounds	        150	            23	            98	            34
>>> 2	variable_1_coeff	12	            18	            3	            2
>>> 3	variable_2_coeff	4	            9	            10	            10
```

### Two Dataframes - Objective Function
The objective function data (including variables and variable coefficients) should only be loaded
according to a single standard.  Variable names should be listed in the first column, lower bounds in the second, upper 
bounds in the third, and coefficients in the fourth.

```python
>>>   variable_names  lower_bounds upper_bounds  coefficient
>>>0  FactA_Standard             0          inf           10
>>>1    FactA_Deluxe             0          inf           15
>>>2  FactB_Standard             0          inf           10
>>>3    FactB_Deluxe             0          inf           15
```

## Other Features for LP and IP Classes
### Naming Constraints
In or-tools, constraints have a .name() property in ortools that calls the automatically generated name of the 
constraint.  While this method has its advantages, it is inconvenient to track constraints by a machine-generated name 
instead of the natural langauge name you would regularly use to refer to it.  

In OR Helper, constraints are contained in a dictionary using the constraint names as keys.  The names of constraints can
be referred to throughout the use of the orhelper object as you would normally use a dictionary.

```python
from orhelper import orhelper
import pandas as pd

# In this example, variables and constraints are split into different worksheets.
horiz_constraints_df = pd.read_excel("test.xlsx", sheet_name="constraints")
hoirz_var_df = pd.read_excel("test.xlsx", sheet_name="variables")

orh = orhelper.LPHorizDfs(
    direction='min', 
    constraint_df=horiz_constraints_df, 
    objective_df=hoirz_var_df
)
for constraint in orh.constraints:
    print(constraint, orh.constraints[constraint])
```

### Handling Infinity and Negative Infinity
When intaking data from Excel spreadsheets or other pandas sources, you may see a number of stand-ins for infinity and 
negative infinity that need to be converted to solver.Infinity().  OR Helper handles solver.Infinity() conversions 
automatically, accepting a variety of inputs.  See the dictionary in LPHelper.handle_infinity() and 
IPHelper.handle_infinity() for the full list of accepted inputs.  I've tried to accommodate the most obvious, but if you
see any I've missed open an issue so I can add them.

### Reduced Costs
Reduced costs represent the opportunity cost of each variable.  They are available as a dictionary, with the variables
as the keys.

```python
from orhelper import orhelper
import pandas as pd

# In this example, variables and constraints are split into different worksheets.
horiz_constraints_df = pd.read_excel("test.xlsx", sheet_name="constraints")
hoirz_var_df = pd.read_excel("test.xlsx", sheet_name="variables")

orh = orhelper.LPHorizDfs(
    direction='min', 
    constraint_df=horiz_constraints_df, 
    objective_df=hoirz_var_df
)
for lp_variable in orh.reduced_costs:
    print(lp_variable, orh.reduced_costs[lp_variable])
```

### Dual Values
The dual value (or shadow price) of a variable is the amount the result would change if the variable changes by a single
unit.  They are available as a dictionary, with the variables as the keys.

```python
from orhelper import orhelper
import pandas as pd

# In this example, variables and constraints are split into different worksheets.
horiz_constraints_df = pd.read_excel("test.xlsx", sheet_name="constraints")
hoirz_var_df = pd.read_excel("test.xlsx", sheet_name="variables")

orh = orhelper.LPHorizDfs(
    direction='min', 
    constraint_df=horiz_constraints_df, 
    objective_df=hoirz_var_df
)
for lp_variable in orh.dual_values:
    print(lp_variable, orh.dual_values[lp_variable])
```

## Min and Max Flow Problems
Max and min flow problems are also supported, however these problem types only accept numeric datatypes as integers.  
Because of this, it's best to use numpy as numpy is much faster.  For sake of convenience, you may use these classes to
automatically load DataFrames into max flow and min flow solvers.  At this time only a single structure of 
DataFrames is accepted, as it seems unnatural to structure data any other way for these problems (unlike linear
programming and integer programming).  

### Min Cost Flow

The class orhelper.MinFlow() requires two DataFrames: a graph DataFrame and a supplies Dataframe.

The graph DataFrame holds the start nodes in the first column, the end nodes in the second column, the capacities in the
third column, and the unit costs in the fourt column.

```python
>>>   start_nodes  end_nodes  capacities  unit_costs
>>>0            0          1          15           4
>>>1            0          2           8           4
>>>2            1          2          20           2
>>>3            1          3           4           2
>>>4            1          4          10           6
>>>5            2          3          15           1
>>>6            2          4           4           3
>>>7            3          4          20           2
>>>8            4          2           5           3
```

The supplies DataFrame holds the value of the supply at each node in the first column.  The index corresponds to the
node that the supplies is assigned to.
```python
>>>   supplies
>>>0        20
>>>1         0
>>>2         0
>>>3        -5
>>>4       -15
```

### Max Flow
The class orhelper.MaxFlow() only requires a single DataFrame.  The start nodes should be put in the first column, the
end nodes in the second, and capacities in the third column.

```python
>>>   start_nodes  end_nodes  capacities
>>>0            0          1          20
>>>1            0          2          30
>>>2            0          3          10
>>>3            1          2          40
>>>4            1          4          30
>>>5            2          3          10
>>>6            2          4          20
>>>7            3          2           5
>>>8            3          4          20
```
