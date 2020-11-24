import os
import sys
from   dataclasses import dataclass, field
from   pandas import DataFrame, Series
from   typing      import Tuple
from   typing      import Dict
from   typing      import List
import pandas as pd
import numpy as np

from  functools import reduce
from  scipy.integrate import quad

from  . sbt_util         import f_multiply
from  . sbt_molecules    import FBI, Molecule

from  .  system_of_units import *
import matplotlib.pyplot as plt


known_filters = ["ZT405", "DMLP425", "DMS490","FB430",
                 "FB420", "FES450","NF405"]
F_filters     = ["FB430", "FB420","NF405"]
R_Filters     = ['DMLP425', 'DMS490']

photon = 1
molecule = 1
gp = 0.664

mum2     = mum * mum
GM       = 1e-50 * cm2*cm2*second / (photon * molecule)
us       = 1 / second
ucm      = 1 / cm
umm      = 1 / mm
umum     = 1 / mum
ucm2     = 1 / cm2
ucm3     = 1 / cm3
mW       = milliwatt
Avogadro = 6.023E+23


@dataclass
class SolutionSample:
    name          : str
    concentration : float
    volume        : float

    @property
    def n_molecules(self)->float:
        return Avogadro * self.concentration * self.volume

    @property
    def rho_molecules(self)->float:
        return self.Avogadro * self.concentration


    def __repr__(self):
        s ="""
        Solution  name ={0};
        concentration = {1:5.1e} mole/l ({2:5.1e} molecules/cm3);
        V = {3:5.1e} l,
        nof molecules = {4:5.1e}
        """.format(self.name,
                   self.concentration/(mole/l),
                   self.rho_molecules/(1/cm3),
                   self.volume/l,
                   self.n_molecules)

        return s


    def __str__(self):
        return self.__repr__()


@dataclass
class Laser:
    lamda  : float
    power  : float

    @property
    def photon_energy(self) -> float:
        lnm = self.lamda / nm
        return (1240 / lnm) * eV

    def energy(self, time : float) -> float:
        return self.power * time

    @property
    def n_photons(self):
        return self.power / self.photon_energy

    def __repr__(self):
        s =f"""
        Laser:
        wavelength                ={self.lamda/nm:5.1e} nm
        photon energy             ={self.photon_energy/eV:5.1e} eV
        power                     ={self.power/milliwatt:5.1e} mW
        photons per second        ={self.n_photons / us:5.1e} ph/second
        """
        return s

    def __str__(self):
        return self.__repr__()


@dataclass
class Objective:
    numerical_aperture : float
    magnification      : float

    def transmission_na(self, A)->float:
        return (1 - np.sqrt(1 - A**2)) /2 if A <=1 else 0.5


    @property
    def transmission(self)->float:
        A = self.numerical_aperture
        return (1 - np.sqrt(1 - A**2)) /2 if A <=1 else 0.5


    def __repr__(self):
        s =f"""
        Objective
        NA                   = {self.numerical_aperture:5.1f}
        Magnification        = {self.magnification:5.1f}
        Transmission         = {self.transmission:5.2e}
        """
        return s


    def __str__(self):
        return self.__repr__()


    def plot_transmission_na(self, figsize=(8,8)):
        """Plots the objective efficiency
        (a function of the NA)
        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        AA=np.linspace(0., 1., 100)
        tna = [self.transmission_na(a) for a in AA]
        plt.plot(AA, tna)
        plt.xlabel(r'NA')
        plt.ylabel(r' transmission ')
        plt.show()


@dataclass
class CCD:
    name             : str = "C13440-20CU"
    n_pixels         : ()  = (2048, 2048)
    size_pixels      : ()  = (6.5 * micron, 6.5 * micron)
    effective_area   : ()  = (13.312 * mm, 13.312 * mm)
    linear_full_well : ()  = (3.0E+5, 1.5E+5) # electrons
    pixel_clock_rate : ()  = (85 * MHZ, 11 * MHZ, 0.6875 * MHZ)
    dark_current     : float  = 0.06 # electron/pixel/s
    readout_noise    : float  = 1.0 # electron
    readout_speed    : float  = 100 # frames/s
    k                : float  = 0.46 # electron/count

    @property
    def pixels (self)->float:
        return self.n_pixels[0] * self.n_pixels[1]


    def efficiency(self, lamda : float)->float:
        xp = np.array([350, 400,450,500,550,600,650,700,750,800,850,
                      900,950,1000])
        fp = np.array([0.3, 0.4,0.65,0.78,0.82,0.82,0.8,0.72,0.62,0.5,0.37,
                      0.24,0.12,0.07])
        return np.interp(lamda/nm, xp, fp)



    def __repr__(self):
        sp = np.array(self.size_pixels) / micron
        ea = np.array(self.effective_area) / mm
        s =f"""
        CCD
        n_pixels             = {self.n_pixels}
        size_pixels          = {sp:5.2e}
        effective_area       = {self.effective_area:5.2e}
        efficiency at 450 nm = {self.efficiency(450*nm):5.2e}
        """
        return s


    def __str__(self):
        return self.__repr__()


    def plot_efficiency(self, figsize=(8,8)):
        """Plots the CCD efficiency

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(350*nm, 1000*nm, 500)
        plt.plot(LAMBDA/nm, self.efficiency(LAMBDA))
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r' efficiency ')
        plt.show()


@dataclass
class Filter:
    """Describes the response of a filter (or dichroic)"""
    name      : str  #filter name: Filter data in specific folder

    def __post_init__(self):
        """Read data from file"""

        if self.name not in known_filters:
            print(f"Filter name not known")
            sys.exit()

        if self.name == "ZT405":
            self.fdf = pd.pandas.read_csv(
            f"{os.environ['SABATDATA']}/FILTERS/ZT405_532rpc-UF1.txt", sep='\t')
        else:
            self.fdf = pd.read_excel(
            f"{os.environ['SABATDATA']}/FILTERS/{self.name}.xlsx")


    def transmission(self, lamda : float)->float:
        """ Returns the filter transmission """

        if self.name in F_filters:
            W  = np.flip(self.fdf.W.values)
            Tr = np.flip(self.fdf.Tr.values/100)
        else:
            W  = self.fdf.W.values
            Tr = self.fdf.Tr.values/100
        return np.interp(lamda/nm, W, Tr)


    def reflection(self, lamda : float)->float:
        """ Returns the filter transmission """

        try:
            R = self.fdf.R.values/100
        except AttributeError:
            print(f'** error, filter does not have reflection data')
            return 0

        if self.name in F_filters:
            W  = np.flip(self.fdf.W.values)
            R = np.flip(R)
        else:
            W  = self.fdf.W.values
            #R = self.fdf.R.values/100
        return np.interp(lamda/nm, W, R)


    def plot_filter_response(self, li, lu, T=True, R=False, figsize=(8,8)):
        """Plots the tranmission between wavelengths li and lu

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)
        if T:
            plt.plot(LAMBDA/nm, self.transmission(LAMBDA), label='Transmission')
            plt.ylabel(r'Transmission')

        if R and self.name in R_Filters:
            plt.plot(LAMBDA/nm, self.reflection(LAMBDA), label='Reflection')
            plt.ylabel(r'Reflection')

        plt.xlabel(r'$\lambda$ (nm)')
        plt.legend()
        plt.title(self.name)
        plt.show()


@dataclass
class FbiEff:
    """Selection efficiency for FBI and FBI+Ba"""
    fbi_slc   : float # efficiency for FBI in Silice
    fbiba_slc : float # efficiency for FBI+Ba
    fbi_sol   : float # efficiency for FBI in Solution
    fbiba_sol : float # efficiency for FBI+Ba

    @property
    def to_list(self):
        return [self.fbi_slc, self.fbiba_slc, self.fbi_sol, self.fbiba_sol ]


@dataclass
class Setup:
    """Laboratory setup"""
    fbi        :  Molecule
    laser      :  Laser
    objective  :  Objective
    ccd        :  CCD
    filters    :  List[Filter]

    def __post_init__(self):
        """Compose filter function and compute normalising integrals"""

        Tr =[f.transmission for f in self.filters]
        self.F = reduce(f_multiply, Tr)

        self.Ifbi_slc, _   = quad(self.fbi.fbi_slc, 350*nm, 750*nm,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        self.IfbiBa_slc, _ = quad(self.fbi.fbi_ba_slc, 350*nm, 750*nm,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)

        self.Ifbi_sol, _   = quad(self.fbi.fbi_sol, 350*nm, 750*nm,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        self.IfbiBa_sol, _ = quad(self.fbi.fbi_ba_sol, 350*nm, 750*nm,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)


    def fbi_slc_t(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in silica transmitted after the filters of the setup.

        """
        return self.fbi.fbi_slc(lamda) * self.F(lamda)


    def fbi_ba_slc_t(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI_Ba2)
        in silica transmitted after the filters of the setup.

        """
        return self.fbi.fbi_ba_slc(lamda) * self.F(lamda)


    def fbi_sol_t(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in solution transmitted after the filters of the setup.

        """
        return self.fbi.fbi_sol(lamda) * self.F(lamda)


    def fbi_ba_sol_t(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI_Ba2)
        in solution transmitted after the filters of the setup.

        """
        return self.fbi.fbi_ba_sol(lamda) * self.F(lamda)


    def fbi_slc_ccd(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in silica transmitted after the filters and the CCD
        of the setup.

        """
        return self.fbi_slc_t(lamda) * self.ccd.efficiency(lamda)


    def fbi_ba_slc_ccd(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI_Ba2)
        in silica transmitted after the filters and the CCD
        of the setup.

        """
        return self.fbi_ba_slc_t(lamda) * self.ccd.efficiency(lamda)


    def fbi_sol_ccd(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI unchelated (FBI)
        in solution transmitted after the filters and the CCD
        of the setup.

        """
        return self.fbi_sol_t(lamda) * self.ccd.efficiency(lamda)


    def fbi_ba_sol_ccd(self, lamda : float)->float:
        """ Returns the emission spectrum for FBI chelated (FBI_Ba2)
        in solution transmitted after the filters and the CCD
        of the setup.

        """
        return self.fbi_ba_sol_t(lamda) * self.ccd.efficiency(lamda)


    def filter_efficiency_fbi(self, lmin : float, lmax : float)->FbiEff:
        """Computes the filter efficiency between lmin and lmax
        for fbi and fbiba

        """

        Tfbi_slc, _   = quad(self.fbi_slc_t, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        TfbiBa_slc, _ = quad(self.fbi_ba_slc_t, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)

        Tfbi_sol, _   = quad(self.fbi_sol_t, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        TfbiBa_sol, _ = quad(self.fbi_ba_sol_t, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)

        eff_fbi_slc       =  Tfbi_slc / self.Ifbi_slc
        eff_fbiba_slc     =  TfbiBa_slc / self.IfbiBa_slc
        eff_fbi_sol       =  Tfbi_sol / self.Ifbi_sol
        eff_fbiba_sol     =  TfbiBa_sol / self.IfbiBa_sol

        return FbiEff(eff_fbi_slc, eff_fbiba_slc, eff_fbi_sol, eff_fbiba_sol)


    def filter_efficiency_lamda(self, lamda : List[float])->List[float]:
        """Computes the filter efficiency for an array lamda of wavelengths"""

        return [self.F(l) for l in lamda]


    def filter_ccd_efficiency_fbi(self, lmin : float, lmax : float)->FbiEff:
        """Computes the filter and CCD efficiency between lmin and lmax
        for fbi and fbiba

        """
        Tfbi_slc, _   = quad(self.fbi_slc_ccd, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        TfbiBa_slc, _ = quad(self.fbi_ba_slc_ccd, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)

        Tfbi_sol, _   = quad(self.fbi_sol_ccd, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)
        TfbiBa_sol, _ = quad(self.fbi_ba_sol_ccd, lmin, lmax,
                                    epsabs=1e-6, epsrel=1e-06, limit=100)

        eff_fbi_slc       =  Tfbi_slc / self.Ifbi_slc
        eff_fbiba_slc     =  TfbiBa_slc / self.IfbiBa_slc
        eff_fbi_sol       =  Tfbi_sol / self.Ifbi_sol
        eff_fbiba_sol     =  TfbiBa_sol / self.IfbiBa_sol

        return FbiEff(eff_fbi_slc, eff_fbiba_slc, eff_fbi_sol, eff_fbiba_sol)


    def setup_efficiency_fbi(self, lmin : float, lmax : float)->FbiEff:
        """Computes the overall efficiency of the setup for FBI
        and FBI+BA. The efficiency is the product of the efficiency
        of the filters, the obective and the CCD

        """
        eff_f = self.filter_ccd_efficiency_fbi(lmin, lmax).to_list
        eff_obj = self.objective.transmission
        eff = [ef *  eff_obj for ef in eff_f]
        return FbiEff(*eff)


    def __repr__(self):
        f_names =[f.name for f in self.filters]
        s =f"""
        Setup:
        FBI        = {self.fbi}
        laser      = {self.laser}
        objective  = {self.objective}
        ccd        = {self.ccd}
        filters    = {f_names}
        """
        return s


    def __str__(self):
        return self.__repr__()


    def plot_fbi_slc_t(self, li, lu, display='both', figsize=(8,8)):
        """Plots the tranmission between wavelengths li and lu

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)

        if display == 'fbi' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_slc_t(LAMBDA), label='FBI in Silica + setup')
        if display == 'fbi-ba' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_ba_slc_t(LAMBDA), label='FBI+BA2+ in Silica + setup')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (a.u)')
        plt.legend()
        plt.show()


    def plot_fbi_sol_t(self, li, lu, display='both', figsize=(8,8)):
        """Plots the tranmission between wavelengths li and lu

        """
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(1, 1, 1)
        LAMBDA  = np.linspace(li, lu, 500)

        if display == 'fbi' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_sol_t(LAMBDA),
                     label='FBI in Solution + setup')
        if display == 'fbi-ba' or display == 'both':
            plt.plot(LAMBDA/nm, self.fbi_ba_sol_t(LAMBDA),
                      label='FBI+BA2+ in Solution  + setup')
        plt.xlabel(r'$\lambda$ (nm)')
        plt.ylabel(r'$\sigma$ (a.u)')
        plt.legend()
        plt.show()
