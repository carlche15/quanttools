import numpy as np
import pandas as pd

def winsorize(data, **kwargs):

    query = ""
    for name, value in kwargs.items():
        assert (type(value) is tuple and value[1] > value[0]), "winsorization lb and ub should be a tuple (lb,ub)"
        lb = np.percentile(data[name], value[0])
        ub = np.percentile(data[name], value[1])
        query += f" {name}>{lb} and {name}<{ub} and"
    return data.query(query[:-3])  # drop the last "and" from the query

