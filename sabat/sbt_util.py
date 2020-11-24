import numpy as np
import pandas as pd

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import Any
from typing      import Optional
from typing      import Callable


from dataclasses import dataclass
from pandas import DataFrame, Series

def pd_find_location(df : DataFrame, column_name, column_value : Any)->Series:
    """Given a DataFrame df, this function finds the occurrence of a value
    column_value in a column with name column_name and return the series
    corresponding to the horizontal slice

    """

    ii = df[column_name].values               # get the column as numpy array
    l  = np.where(ii == column_value)[0][0]   # get the location of the first
                                              # occurence of value
    return df.iloc[l]                          # return the series


def select_in_range(df, varx, xmin, xmax):
    mask = in_range(df[varx], xmin, xmax)
    return df[mask]


def f_multiply(f : Callable, g : Callable):
    return lambda x: f(x) * g(x)
