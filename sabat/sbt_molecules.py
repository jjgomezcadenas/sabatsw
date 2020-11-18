import os
from   dataclasses import dataclass
from   pandas import DataFrame
import pandas as pd
import numpy as np
from  . sbt_util     import pd_find_location
from  . sbt_analysis import subtract_spectra
from  . sbt_analysis import scale_spectra
from  . sbt_analysis import loc_w

from .  system_of_units import *
import matplotlib.pyplot as plt

GM = 1e-50 * cm2*cm2*second

def combine_dfs_(df1 : DataFrame, df2 : DataFrame,
                 wc : float, w1 : float =1, w2 : float =1)->DataFrame:
    """Conbines df1 and df2
    The resultins data frame is:
     w1 * df1  for w < wc
     w2 * df2  for w > wc
     Where weights w1 and w2 are empirically determined

    """
    ic  = loc_w(df1, wc)
    dfa = df1[0:ic].copy()
    dfb = df2[ic:-1].copy()

    dfa['I'] = dfa['I'].map(lambda x: x * w1)
    dfb['I'] = dfb['I'].map(lambda x: x * w2)

    return pd.concat([dfa,dfb])


def get_WI_from_df_(df : DataFrame, I : str)->DataFrame:
    """Prepares a standard W,I,sI DataFrame from values
    described in column I
    W  : wavelength in nm
    I  : Intensity in arbitrary units
    si : error on intensity

    """
    w  = df['W']
    i  = df[I]
    si = np.sqrt(abs(i))

    return pd.concat([w,i,si], axis=1, keys=['W','I','SI'])


def fbi_ba2_slc_emission(scale : float =0.9)->DataFrame:
    """ Computes the emission spectrum of fbi-Ba2 in silica (slc)
    from data.
    FBI250.xlsx    = fluorimeter data describing response to FBI+Ba
                  in silica

    silice250.xlsx = response of the silica
    The scale is an empiric parameter which adjustes the baselines

    """
    fbi250df = pd.read_excel(f"{os.environ['SABATDATA']}/FBI250.xlsx")
    slc250df = pd.read_excel(f"{os.environ['SABATDATA']}/silice250.xlsx")

    # Column I describes the specific data set
    # In both cases the excitation light was 250 nm

    silBAF5  = get_WI_from_df_(fbi250df, I='BAF5_250')
    slc250   = get_WI_from_df_(slc250df, I='SLC_250_I')
    return subtract_spectra(silBAF5, scale_spectra(slc250,scale))


def fbi_slc_emission(scale1 : float =0.36, scale2 : float = 0.6)->DataFrame:
    """ Computes the emission spectrum of fbiin silica (slc)
    from data.
    FBI_POWDER_20_07_19.xlsx  = fluorimeter data describing response to FBI
                                and also silica

    18caFBISilice250nm.xlsx   = response of silica

    The response of silica is calibrated through the empirical values
    of scale1 and scale 2

    """

    slcFbi250df = \
    pd.read_excel(f"{os.environ['SABATDATA']}/FBI_POWDER_20_07_19.xlsx")

    silA250df = \
    pd.pandas.read_csv(f"{os.environ['SABATDATA']}/silice.csv",
                       dtype={"W": np.float64, "I": np.float64, "c": str})

    sil250df = \
    pd.read_excel(f"{os.environ['SABATDATA']}/18caFBISilice250nm.xlsx")

    slcFbi250D = get_WI_from_df_(slcFbi250df, I='POWDER_B_250')
    sil250     = get_WI_from_df_(sil250df,    I='Silice')
    silA250    = get_WI_from_df_(silA250df,   I='I')

    sb250      = combine_dfs_(sil250, silA250, wc = 400, w1 = 0.55, w2 = 0.32)

    return subtract_spectra(scale_spectra(slcFbi250D,scale1),
                                scale_spectra(sb250,scale2))

@dataclass
class FBIsPA:
    """FBI single Photon Absorption cross section"""
    l_peak    : float = 250         # lambda value for which sigma specified
                                    # in nm
    s_fbi     : float = 0.064 * nm2 # value of the cross section for FBI
    s_fbi_ba  : float = 0.068 * nm2 # value of the cross section for FBI_BA


    def __post_init__(self):
        """Read data from file:
        EDI_029_absorption.xlsx : absorption cross section measured by
                                  fluorimeter in arbitrary system_of_units
        FBI250.xlsx             : emission spectrum on a silica pellet
        silice250.xlsx          : silica background

        """

        self.fbi_abs_spectrum()
        self.fbiBa2 = fbi_ba2_slc_emission(scale=0.9)
        self.fbi   = fbi_slc_emission(scale1=0.36, scale2 = 0.6)


    def fbi_abs_spectrum(self):
        self.fbidf    = \
        pd.read_excel(f"{os.environ['SABATDATA']}/EDI_029_absorption.xlsx")
        s = pd_find_location(self.fbidf, "II", self.l_peak)
        sfbi_ba_au      = s['FBI_Ba']
        sfbi_au         = s['FBI']
        self.sfbi_ba_nm = self.s_fbi_ba / sfbi_ba_au
        self.sfbi_nm    = self.s_fbi / sfbi_au


    def fbi_ba_sigma(self, lamda : float)->float:
        """ Returns the absorption cross section for FBI chelated (FBI+Ba)"""
        return np.interp(lamda/nm, self.fbidf.II.values,
                         self.fbidf.FBI_Ba.values) * self.sfbi_ba_nm


    def fbi_sigma(self, lamda : float)->float:
        """ Returns the absorption cross section for FBI unchelated (FBI)"""
        return np.interp(lamda/nm, self.fbidf.II.values,
                         self.fbidf.FBI.values) * self.sfbi_nm


    def plot_sigma(self, li, lu, figsize=(8,8)):
        """Plots the cross section between wavelengths li and lu"""
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        plt.plot(LAMBDA/nm, self.fbi_sigma(LAMBDA)/nm2, label='FBI')
        plt.plot(LAMBDA/nm, self.fbi_ba_sigma(LAMBDA)/nm2, label='FBI+BA')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (nm$^2$)')
        plt.legend()
        plt.show()
