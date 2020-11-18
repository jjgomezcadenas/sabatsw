"""
Tests for sbt_molecules
"""

import numpy          as np
import pandas as pd

from  sabat.sbt_util   import pd_find_location


def test_pd_find_location():
    rownames = ['row1', 'row2', 'row3', 'row4', 'row5']
    colnames = ['col1', 'col2', 'col3', 'col4']

    # Create a 5 row by 4 column array of integers from 0 to 19
    integers = np.arange(20).reshape((5, 4))
    table = pd.DataFrame(integers, index=rownames, columns=colnames)

    j=0
    for i in range(0,20,4):
        s = pd_find_location(table, 'col1', i)
        assert s['col1'] == j
        assert s['col2'] == j + 1
        assert s['col3'] == j + 2
        assert s['col4'] == j + 3
        j+=4
