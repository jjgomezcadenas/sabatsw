"""
Tests for sbt_molecules
"""

import numpy          as np
from   numpy          import allclose

from  sabat.sbt_molecules   import FBIsPA
from  sabat.system_of_units import *

def test_fbispa():
    fbisp = FBIsPA()
    nfbic = fbisp.fbidf.FBI_Ba.values * fbisp.sfbi_ba_nm
    nfbiu = fbisp.fbidf.FBI.values    * fbisp.sfbi_nm

    ifbic = fbisp.fbi_ba_sigma(fbisp.fbidf.II.values*nm)/nm2
    ifbiu = fbisp.fbi_sigma(fbisp.fbidf.II.values*nm)/nm2

    assert allclose(nfbic/nm2, ifbic)
    assert allclose(nfbiu/nm2, ifbiu)
