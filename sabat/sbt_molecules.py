import os
from   dataclasses import dataclass, field
from   pandas import DataFrame, Series
from   typing      import Tuple
from   typing      import Dict
from   typing      import List
import pandas as pd
import numpy as np
from  . sbt_util     import pd_find_location
from  . sbt_analysis import subtract_spectra
from  . sbt_analysis import scale_spectra
from  . sbt_analysis import loc_w

from .  system_of_units import *
from invisible_cities.core.core_functions import in_range
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


def fbi_ba2_sol_emission()->DataFrame:
    """ Computes the emission spectrum of fbi-Ba2 in solution (sol)
    from data.
    FBI_solution.xlsx    = fluorimeter data describing response to FBI+Ba
                           in solution

    """
    fbi250df = pd.read_excel(f"{os.environ['SABATDATA']}/FBI_solution.xlsx")

    # Column I describes the specific data set
    # In both cases the excitation light was 250 nm

    solBA  = get_WI_from_df_(fbi250df, I='FBI_Ba')
    return subtract_spectra(silBAF5, scale_spectra(slc250,scale))


@dataclass
class Molecule:
    Q  : float = 0.9  # quantum efficiency


@dataclass
class FBI(Molecule):
    """Describes the response of the FBI molecule in Silica
    and in Solution

    """
    norm      : bool = True
    # lambda ranges of interest, each range surrounds a local maxima
    # maxima are located in different places for FBI and FBI-Ba

    lrfbi    : List = field(default_factory=lambda: [(230,290),
                                                    (290, 320),
                                                    (320, 370),
                                                    (370, 470)])

    lrfbi_ba : List = field(default_factory=lambda: [(230,310),
                                                     (310, 350),
                                                     (350, 410),
                                                     (410, 470)])

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
        self.fbi_slc_spectrum()
        self.fbi_sol_spectrum()


    def fbi_abs_spectrum(self):
        """Compute the absorption spectrum and related variables"""

        self.fbidf    = \
        pd.read_excel(f"{os.environ['SABATDATA']}/EDI_029_absorption.xlsx")

        # normalise to the value of the cross section in l_peak
        if self.norm:
            s = pd_find_location(self.fbidf, "II", self.l_peak)
            sfbi_ba_au           = s['FBI_Ba']
            sfbi_au              = s['FBI']
            self.sfbi_ba_nm      = self.s_fbi_ba / sfbi_ba_au
            self.sfbi_nm         = self.s_fbi / sfbi_au
        else:
            self.sfbi_ba_nm      = 1
            self.sfbi_nm         = 1

        # Define Sector data frames according to lrfbi and lrfbi_ba
        # These are portions of the spectrum that contained a single maximum
        fbiS   =[self.fbidf[in_range(self.fbidf.II,
                                     *self.lrfbi[i])] for i in range(4)]
        fbibaS =[self.fbidf[in_range(self.fbidf.II,
                                     *self.lrfbi_ba[i])] for i in range(4)]

        # Store a DF with W, FBI, FBI_Ba data for each sector
        self.max_fbi    = [pd_find_location(fbi, 'FBI',
                                        fbi['FBI'].max()) for fbi in fbiS ]

        self.max_fbi_ba = [pd_find_location(fbi, 'FBI_Ba',
                                        fbi['FBI_Ba'].max()) for fbi in fbibaS ]

    def fbi_slc_spectrum(self):
        """Compute the emission spectrum in silica (slc)
        and related variables

        """
        self.fbiSlcBa2df    = fbi_ba2_slc_emission(scale=0.9)
        self.fbiSlcdf       = fbi_slc_emission(scale1=0.36, scale2 = 0.6)

        if self.norm:
            self.fbi_slc_norm     = self.fbiSlcdf.I.sum()
            self.fbi_ba2_slc_norm = self.fbiSlcBa2df.I.sum()
        else:
            self.fbi_slc_norm     = 1
            self.fbi_ba2_slc_norm = 1

        self.max_fbi_slc_ba = pd_find_location(self.fbiSlcBa2df, 'I',
                                               self.fbiSlcBa2df['I'].max())
        self.max_fbi_slc    = pd_find_location(self.fbiSlcdf, 'I',
                                               self.fbiSlcdf['I'].max())


    def fbi_sol_spectrum(self):
        """Compute the emission spectrum in solution (sol)
        and related variables

        """
        fbi250df = pd.read_excel(f"{os.environ['SABATDATA']}/FBI_solution.xlsx")
        self.fbiSolBa2df  = get_WI_from_df_(fbi250df, I='FBI_Ba')
        self.fbiSoldf     = get_WI_from_df_(fbi250df, I='FBI')

        if self.norm:
            self.fbi_sol_norm     = self.fbiSoldf.I.sum()
            self.fbi_ba2_sol_norm = self.fbiSolBa2df.I.sum()
        else:
            self.fbi_sol_norm     = 1
            self.fbi_ba2_sol_norm = 1

        self.max_fbi_sol_ba = pd_find_location(self.fbiSolBa2df, 'I',
                                               self.fbiSolBa2df['I'].max())
        self.max_fbi_sol    = pd_find_location(self.fbiSoldf, 'I',
                                           self.fbiSoldf['I'].max())


    @property
    def abs_maxima(self)->DataFrame:
        """Returns a Data Frame with all absorption maxima"""
        w      = self.max_fbi[0].index.values
        if self.norm:
            n1 = self.sfbi_nm / nm2    # normalise and express in nm2
            n2 = self.sfbi_ba_nm / nm2
        else:
            n1 = 1
            n2 = 1

        # Prepare a list of lists:
        # each element ->[lambda, sigma_fbi, sigma_fbi_ba]
        # The list runs over all the sectors
        # Notice that lambda (in nm) is given by mfgi.II
        # Notice normalisation of cross sections

        max_fbi_ll   = [ [mfbi.II, mfbi.FBI_Ba * n2,
                        mfbi.FBI * n1] for mfbi in self.max_fbi]

        max_fbi_ba_ll = [ [mfbi.II, mfbi.FBI_Ba * n2,
                        mfbi.FBI * n1] for mfbi in self.max_fbi_ba]

        return pd.DataFrame(list(zip(w, *max_fbi_ll, *max_fbi_ba_ll)),
                            columns=['par',
                            'max1_fbi','max2_fbi', 'max3_fbi', 'max4_fbi',
                            'max1_fbi_ba','max2_fbi_ba', 'max3_fbi_ba',
                            'max4_fbi_ba'])


    def fbi_ba_sigma(self, lamda : float)->float:
        """ Returns the absorption cross section for FBI chelated (FBI+Ba)"""
        return np.interp(lamda/nm, self.fbidf.II.values,
                         self.fbidf.FBI_Ba.values) * self.sfbi_ba_nm


    @property
    def fbi_ba_sigma_max(self)->Series:
        """ Returns the max absorption cross section
        for FBI chelated (FBI+Ba)

        """
        return self.fbidf.FBI_Ba.max()


    def fbi_sigma(self, lamda : float)->float:
        """ Returns the absorption cross section for FBI unchelated (FBI)"""
        return np.interp(lamda/nm, self.fbidf.II.values,
                         self.fbidf.FBI.values) * self.sfbi_nm


    def fbi_slc(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in silica.

        """
        return np.interp(lamda/nm, self.fbiSlcdf.W.values,
                         self.fbiSlcdf.I.values / self.fbi_slc_norm)


    def fbi_ba_slc(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI-Ba)
        in silica.

        """
        return np.interp(lamda/nm, self.fbiSlcBa2df.W.values,
                         self.fbiSlcBa2df.I.values / self.fbi_ba2_slc_norm)


    def fbi_sol(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in solution.

        """
        return np.interp(lamda/nm, self.fbiSoldf.W.values,
                         self.fbiSoldf.I.values / self.fbi_sol_norm)


    def fbi_ba_sol(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI-Ba)
        in solution.

        """
        return np.interp(lamda/nm, self.fbiSolBa2df.W.values,
                         self.fbiSolBa2df.I.values / self.fbi_ba2_sol_norm)


    def plot_sigma(self, li, lu, display='both', figsize=(8,8)):
        """Plots the cross section between wavelengths li and lu
           display can take values: fbi, fbi-ba or both

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        if self.norm:
            if display == 'fbi' or display == 'both':
                plt.plot(LAMBDA/nm, self.fbi_sigma(LAMBDA)/nm2, label='FBI')
            if display == 'fbi-ba' or display == 'both':
                plt.plot(LAMBDA/nm, self.fbi_ba_sigma(LAMBDA)/nm2, label='FBI+BA')
            plt.ylabel(r'$\sigma$ (nm$^2$)')
        else:
            if display == 'fbi' or display == 'both':
                plt.plot(LAMBDA/nm, self.fbi_sigma(LAMBDA), label='FBI')
            if display == 'fbi-ba' or display == 'both':
                plt.plot(LAMBDA/nm, self.fbi_ba_sigma(LAMBDA), label='FBI+BA')
            plt.ylabel(r'$\sigma$ (a.u.)')

        plt.xlabel(r'$\lambda$ (nm)')
        plt.legend()
        plt.show()


    def plot_fbi_slc(self, li, lu, display='both', figsize=(8,8)):
        """Plots the cross section between wavelengths li and lu
           display can take values: fbi, fbi-ba or both

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        if display == 'fbi' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_slc(LAMBDA), label='FBI in Silica')
        if display == 'fbi-ba' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_ba_slc(LAMBDA), label='FBI+BA2+ Silica')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (a.u)')
        plt.legend()
        plt.show()


    def plot_fbi_sol(self, li, lu, display='both', figsize=(8,8)):
        """Plots the cross section between wavelengths li and lu
           display can take values: fbi, fbi-ba or both

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        if display == 'fbi' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_sol(LAMBDA), label='FBI in Solution')
        if display == 'fbi-ba' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_ba_sol(LAMBDA), label='FBI+BA2+ Solution')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (a.u)')
        plt.legend()
        plt.show()


    def plot_fbi_sil_sol(self, li, lu, display='both', figsize=(8,8)):
        """Plots the cross section between wavelengths li and lu
           display can take values: fbi, fbi-ba or both

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        if display == 'fbi' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_slc(LAMBDA), label='FBI in Silica')
            plt.plot(LAMBDA/nm, self.fbi_sol(LAMBDA), label='FBI in Solution')
        if display == 'fbi-ba' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_ba_slc(LAMBDA), label='FBI+BA2+ Silica')
            plt.plot(LAMBDA/nm, self.fbi_ba_sol(LAMBDA), label='FBI+BA2+ Solution')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (a.u)')
        plt.legend()
        plt.show()


    def __repr__(self):
        self.max_fbi_slc["W"]
        s =f"""
        FBI properties:
        Q                                ={self.Q}
        lambda peak (FBI) in Silica      ={self.max_fbi_slc["W"]:5.1f} nm
        lambda peak (FBI+Ba) in Silica   ={self.max_fbi_slc_ba["W"]:5.1f} nm
        lambda peak (FBI) in Solution    ={self.max_fbi_sol["W"]:5.1f} nm
        lambda peak (FBI+Ba) in Solution ={self.max_fbi_sol_ba["W"]:5.1f} nm

        """
        return s


    def __str__(self):
        return self.__repr__()
