from   dataclasses import dataclass
from   typing      import Dict
from   typing      import TypeVar

import numpy as np
from   invisible_cities.core.system_of_units import *

@dataclass
class FoV:
    x : float
    y : float
    z : float

    def area(self)->float:
        return self.x * self.y

    def volume(self)->float:
        return self.x * self.y * self.z

    def __repr__(self):
        s ="""
        FOV; x = {0:5.1e} mm; y = {1:5.1e} mm; z = {2:5.1e} mm; area = {3:5.1e} mm2 volume = {4:5.1e} mm3
        """.format(self.x/mm, self.y/mm, self.z/mm, self.area()/mm2, self.volume()/mm3)

        return s


@dataclass
class Laser:
    lamda  : float
    power  : float

    def photon_energy(self) -> float:
        lnm = self.lamda / nm
        return (1240 / lnm) * eV

    def energy(self, time : float) -> float:
        return self.power * time

    def n_photons(self):
        return self.power / self.photon_energy()

    def __repr__(self):
        s ="""
        Laser:
        wavelength                ={0:5.1e} nm
        photon energy             ={1:5.1e} eV
        power                     ={2:5.1e} mW
        energy per second         ={6:5.1e} mJ
        photons per second        ={6:5.1e} ph/second
        """.format(self.lamda/nm,
                   self.photon_energy()/eV,
                   self.power/milliwatt,
                   self.f/MHz,
                   self.tau/fs,
                   self.energy(1*second)/mJ,
                   self.n_photons() / (1/second)
                   )
        return s


@dataclass
class PulsedLaser(Laser):
    f      : float
    tau    : float

    def energy_per_pulse(self) -> float:
        return self.power * self.f


    def __repr__(self):
        s ="""
        Pulsed Laser:
        wavelength                ={0:5.1e} nm
        photon energy             ={1:5.1e} eV
        power                     ={2:5.1e} mW
        repetition rate           ={3:5.1e} kHz
        pulse width               ={4:5.1e} fs
        energy per pulse          ={5:5.1e} fJ
        energy per second         ={6:5.1e} mJ
        photons per second        ={6:5.1e} ph/second
        """.format(self.lamda/nm,
                   self.photon_energy()/eV,
                   self.power/milliwatt,
                   self.f/MHz,
                   self.tau/fs,
                   self.energy_per_pulse()/fJ,
                   self.energy(1*second)/mJ,
                   self.n_photons() / (1/second)
                   )
        return s

GLaser = TypeVar('GenericLaser',  Laser, PulsedLaser)  #GenericLaser

@dataclass
class Molecule:
    name   : str
    sigma  : float
    Q      : float

    def max_photons(self, case : str)-> float:
        mp ={'oxygenated': 36000,'deoxygenated' : 360000, 'dry' : 1e+30}
        return mp[case]


    def __repr__(self):
        s ="""
        Molecule name ={0}; cross section = {1:5.1e} cm2; Q = {2:5.1f}
        """.format(self.name,
                   self.sigma/cm2,
                   self.Q)
        return s


@dataclass
class Molecule2P(Molecule):
    GM = 1e-50 * cm2*cm2*second
    lamda : np.array = np.array([691 * nm,
                        700 * nm ,
                        720 * nm ,
                        740 * nm ,
                        760 * nm ,
                        780 * nm ,
                        800 * nm ,
                        820 * nm ,
                        840 * nm ,
                        860 * nm ,
                        890 * nm ,
                        900 * nm ,
                        920 * nm ,
                        940 * nm ,
                        960 * nm ,
                        980 * nm ,
                        1000* nm ])
    s2 :    np.array  = np.array([16 * GM,
                         19 * GM,
                         19 * GM,
                         30 * GM,
                         36 * GM,
                         36 * GM,
                         36 * GM,
                         29 * GM,
                         13 * GM,
                         8  * GM,
                         11 * GM,
                         16 * GM,
                         26 * GM,
                         11 * GM,
                         15 * GM,
                         10 * GM,
                         5  * GM])

    def sigma2(self, lamda : float)->float:
        return np.interp(lamda, self.lamda, self.s2)


@dataclass
class DyeSample:
    name          : str
    concentration : float
    volume        : float
    Avogadro      : float = 6.023E+23

    def n_molecules(self)->float:
        return self.Avogadro * self.concentration * self.volume

    def rho_molecules(self)->float:
        return self.Avogadro * self.concentration

    def __repr__(self):
        s ="""
        Dye name ={0};
        concentration = {1:5.1e} mole/l ({1:5.1e} molecules/cm3);
        V = {2:5.1e} l,
        nof molecules = {3:5.1e}
        """.format(self.name,
                   self.concentration/(mole/l),
                   (self.n_molecules()/self.volume)/cm3,
                   self.volume/l,
                   self.n_molecules())

        return s


@dataclass
class Microscope:
    name               : str
    numerical_aperture : float
    magnification      : float
    eff_dichroic       : float = 0.85
    eff_filter         : float = 0.8
    NA                 : float = 1.4
    M                  : float = 100
    T                  : float = 0.3


    def optical_transmission(self)->float:
        f1 = self.numerical_aperture / self.NA
        f2 = self.magnification / self.M
        return self.T * (f1/f2)**2

    def filter_transmission(self)->float:
        return self.eff_dichroic * self.eff_filter

    def transmission(self)->float:
        return self.optical_transmission() * self.filter_transmission()

    def __repr__(self):
        s ="""
        name                 = {0}
        NA                   = {1:5.1f}
        M                    = {2:5.1f}
        eff dichroic         = {3:5.2f}
        eff filter           = {4:5.2f}
        Optical transmission = {5:5.2f}
        Filter  transmission = {6:5.2f}
        Total transmission   = {7:5.2f}
        """.format(self.name,
                   self.numerical_aperture,
                   self.magnification,
                   self.eff_dichroic,
                   self.eff_filter,
                   self.optical_transmission(),
                   self.filter_transmission(),
                   self.transmission())

        return s


@dataclass
class CCD:
    name             : str = "C9100-23B"
    n_pixels         : ()  = (512, 512)
    size_pixels      : ()  = (16 * micron, 16 * micron)
    effective_area   : ()  = (8.19 * mm, 8.19 * mm)
    linear_full_well : ()  = (3.7E+5, 1.5E+5) # electrons
    pixel_clock_rate : ()  = (22 * MHZ, 11 * MHZ, 0.6875 * MHZ)
    dark_current     : float  = 0.005 # electron/pixel/s
    readout_noise    : float  = 8 # electron
    readout_speed    : float  = 72 # frames/s

    def pixels (self)->float:
        return self.n_pixels[0] * self.n_pixels[1]

    def efficiency(self, lamda : np.array)->np.array:
        xp = np.array([300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000])
        fp = np.array([0.1,0.2,0.55,0.8,0.92,0.98,0.98,0.92,0.9,0.8,0.7,0.55,0.4,0.2,0.1])
        return np.interp(lamda/nm, xp, fp)
