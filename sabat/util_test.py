"""
Tests for sbt_molecules
"""

import numpy          as np
import pandas as pd

from  sabat.sbt_util   import pd_find_location
from  sabat.sbt_util   import select_in_range
from  sabat.sbt_util    import f_multiply

from  sabat.sbt_molecules   import FBI
from  sabat.system_of_units import *
from  functools import reduce

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


def select_in_range():
    fbisp = FBI(norm=True) # create a FBI object
    # select the region of response
    fbiBa = select_in_range(fbisp.fbiSlcBa2df,
                            varx="W", xmin=350, xmax=650)
    # its more than 99 % of spectrum
    etot = fbisp.fbiSlcBa2df.sum().I
    xfbi = fbiBa.sum().I / etot
    assert xfbi > 0.99


def test_f_multiply():
    def f1(x):
        return 2*x
    def f2(x):
        return x**2
    def f3(x):
        return x+10
    def test(x):
        return f1(x) * f2(x) * f3(x)
    F = [f1, f2, f3]

    fc = reduce(f_multiply, F)

    for i in range(5):
        assert fc(i) == test(i)
