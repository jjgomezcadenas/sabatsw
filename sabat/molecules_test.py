"""
Tests for sbt_molecules
"""

import numpy          as np
from   numpy          import allclose
from pytest           import approx

from  sabat.sbt_molecules   import FBI
from  sabat.system_of_units import *
from  sabat.sbt_util        import pd_find_location
from invisible_cities.core.core_functions import in_range

def test_fbi_pars():

    fbisp = FBI(Q=0.9)
    assert fbisp.Q                   == approx(0.9)
    assert fbisp.max_fbi_slc["W"]    == approx(509, rel=0.1)
    assert fbisp.max_fbi_slc_ba["W"] == approx(420)
    assert fbisp.max_fbi_sol["W"]    == approx(490)
    assert fbisp.max_fbi_sol_ba["W"] == approx(430)

def test_fbi_sigma():

    fbisp = FBI(norm=True)
    nfbic = fbisp.fbidf.FBI_Ba.values * fbisp.sfbi_ba_nm
    nfbiu = fbisp.fbidf.FBI.values    * fbisp.sfbi_nm

    ifbic = fbisp.fbi_ba_sigma(fbisp.fbidf.II.values*nm)/nm2
    ifbiu = fbisp.fbi_sigma(fbisp.fbidf.II.values*nm)/nm2

    assert allclose(nfbic/nm2, ifbic)
    assert allclose(nfbiu/nm2, ifbiu)


def test_fbi_slc():
    fbisp = FBI(norm=False)

    nfbislc = fbisp.fbiSlcdf['I'].values
    ifbislc = fbisp.fbi_slc(fbisp.fbiSlcdf['W'].values*nm)
    print(nfbislc[0:10])
    print(ifbislc[0:10])
    assert allclose(nfbislc[0:-1], ifbislc[0:-1])


def test_fbi_sol():
    fbisp = FBI(norm=False)

    nfbislc = fbisp.fbiSoldf['I'].values
    ifbislc = fbisp.fbi_sol(fbisp.fbiSoldf['W'].values*nm)
    print(nfbislc[0:10])
    print(ifbislc[0:10])
    assert allclose(nfbislc[0:-1], ifbislc[0:-1])


def test_fbi_ba_slc():
    fbisp = FBI(norm=False)

    nfbislc = fbisp.fbiSlcBa2df['I'].values
    ifbislc = fbisp.fbi_ba_slc(fbisp.fbiSlcBa2df['W'].values*nm)
    print(nfbislc[0:10])
    print(ifbislc[0:10])
    assert allclose(nfbislc[0:-1], ifbislc[0:-1])


def test_fbi_ba_sol():
    fbisp = FBI(norm=False)

    nfbislc = fbisp.fbiSolBa2df['I'].values
    ifbislc = fbisp.fbi_ba_sol(fbisp.fbiSolBa2df['W'].values*nm)
    print(nfbislc[0:10])
    print(ifbislc[0:10])
    assert allclose(nfbislc[0:-1], ifbislc[0:-1])


def test_fbi_slc_norm():
    fbisp = FBI(norm=True)
    w = fbisp.fbiSlcdf.I / fbisp.fbiSlcdf.I.sum()
    assert np.nansum(w.values) == 1


def test_fbi_ba_slc_norm():
    fbisp = FBI(norm=True)
    w = fbisp.fbiSlcBa2df.I / fbisp.fbiSlcBa2df.I.sum()
    assert np.nansum(w.values) == 1


def fbi_conf(fbisp, xnum=0, xmol='FBI'):
    if xmol == 'FBI':
        lr = fbisp.lrfbi
    else:
        lr = fbisp.lrfbi_ba
    df_1 = fbisp.fbidf[in_range(fbisp.fbidf.II, *lr[xnum])]
    df1  = pd_find_location(df_1, xmol, df_1[xmol].max())
    fabm = fbisp.abs_maxima
    assert df1["II"]     == fabm[f'max{xnum+1}_{xmol.lower()}'].loc[0]
    assert df1["FBI_Ba"] == fabm[f'max{xnum+1}_{xmol.lower()}'].loc[1]
    assert df1["FBI"]    == fabm[f'max{xnum+1}_{xmol.lower()}'].loc[2]


def test_fbi_maxima2():
    fbisp = FBI(norm=False)
    for ix in range(4):
        fbi_conf(fbisp, xnum=ix, xmol='FBI')
        fbi_conf(fbisp, xnum=ix, xmol='FBI_Ba')
