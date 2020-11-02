import numpy as np
import pandas as pd
import glob
import os

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar
from typing      import Optional

from enum        import Enum

from dataclasses import dataclass

import matplotlib.pyplot as plt

from . system_of_units import *

from . sbt_types import FoV

rad_to_deg = 180/np.pi
photon = 1
molecule = 1
us = photon / second
ucm  = 1 / cm
umm  = 1 / mm
umum = 1 / mum
mum2 = mum *mum
ucm2 = photon / cm2
ucm3 = molecule / cm3

"""
IOW convention

Bulk (n2)
---------
IOW (n1)
--------
Substrate (n0)

"""

@dataclass
class Adlayer:
    d  : float
    gf : float  # G = nf/area

    @property
    def area(self)->float:
        return np.pi * (self.d/2)**2

    @property
    def nf(self):
        return self.gf * self.area

    def nf_pixel(self, nof_pixels)->float:
        return self.nf / nof_pixels

    def area_pixel(self, nof_pixels)->float:
        return self.area / nof_pixels

    def power_density(self, laser)->float:
        return laser.power / self.area

    def photon_density(self, laser)->float:
        return self.power_density(laser) / laser.photon_energy()

    def __repr__(self):
        s =f"""
        Adlayer:
         d    = {self.d/micron:5.1e} micron
         area = {self.area/mum2:5.1e} mum2
         Gf   = {self.gf/(molecule/nm2):5.1e} molecule/nm2
        """
        return s


@dataclass
class Eigen:
    t1 : float
    t2 : float
    t3 : float
    t4 : float

    def eigen(self)->float:
        return self.t1 - self.t2 - self.t3 - self.t4


@dataclass
class PRoots:
    """Arrays with the solutions of the eigenvalue equations (d, theta)
    where d is the thickness of the IOW and theta the solution angle

    """
    d     : np.array
    theta : np.array

    def __str__(self):
        st = f"Thickness of substrate ={self.d/nm} nm, angles ={self.theta * rad_to_deg} DEG"
        return st

@dataclass
class IOW:
    n0    : float # substrate (e.g, quartz)
    n1    : float # IOW (e.g, ITO)
    n2    : float # bulk (e.g, xenon gas)
    d     : float # thickness of IOW
    L     : float # Lenght of IOW
    lamda : float # wavelength of laser

    @property
    def tc10(self):
        """Angle of total internal reflection between n0 and n1"""
        return np.arcsin(self.n0/self.n1)

    @property
    def tc12(self):
        """Angle of total internal reflection between n2 and n1"""
        return np.arcsin(self.n2/self.n1)

    def d10(self, theta):
        """Distance of penetration from medium n1 into medium n0"""
        return de(self.lamda, self.n0 , self.n1, theta)

    def d12(self, theta):
        """Distance of penetration from medium n1 into medium n2"""
        return de(self.lamda, self.n2 , self.n1, theta)

    def Ie_Ii_TE_10(self,theta):
        """|Ie_Ii_TE_10 = (Ie/Ii)TE between n1 and n0
        TE is the TE polarization
        (Ie/Ii) is the transmitted interfacil intensity
        per unit of incident intensity

        """
        return tTE2(self.n0,self.n1,theta)

    def Ie_Ii_TE_12(self,theta):
        """|Ie_Ii_TE_12 = (Ie/Ii)TE between n1 and n2"""
        return tTE2(self.n2,self.n1,theta)

    def Ie_Ii_TM_10(self,theta):
        """|Ie_Ii_TM_10 = (Ie/Ii)TM between n1 and n0"""
        return tTM2(self.n0,self.n1,theta)

    def Ie_Ii_TM_12(self,theta):
        """|Ie_Ii_TM_12 = (Ie/Ii)TE between n1 and n2"""
        return tTM2(self.n2,self.n1,theta)

    def i12TM(self,z, theta):
        return ie(z,self.lamda,self.n2,self.n1,theta,mode='TM')

    def i10TE(self,z, theta):
        return ie(z,self.lamda,self.n0,self.n1,theta,mode='TE')

    def i12TE(self,z, theta):
        return ie(z,self.lamda,self.n2,self.n1,theta,mode='TE')

    def l10TM(self,theta):
        """Evanescent path length in n0 medium for TM polarization"""
        return le(self.lamda,self.n0,self.n1,theta, mode='TM')

    def l12TM(self,theta):
        """Evanescent path length in n2 medium for TM polarization"""
        return le(self.lamda,self.n2,self.n1,theta, mode='TM')

    def l10TE(self,theta):
        """Evanescent path length in n0 medium for TE polarization"""
        return le(self.lamda,self.n0,self.n1,theta, mode='TE')

    def l12TE(self,theta):
        """Evanescent path length in n2 medium for TE polarization"""
        return le(self.lamda,self.n2,self.n1,theta, mode='TE')

    def dt10TE(self,theta):
        """Goos-Hanchen shifts at the waveguide/substrate interfaces TE polarization """
        return self.l10TE(theta) * (np.cos(theta) * np.sin(theta)) / nij(self.n0, self.n1)

    def dt12TE(self,theta):
        """Goos-Hanchen shifts at the waveguide/superstrate TE polarization """
        return self.l12TE(theta) * (np.cos(theta) * np.sin(theta)) / nij(self.n2, self.n1)

    def dt10TM(self,theta):
        """Goos-Hanchen shifts at the waveguide/superstrate TM polarization """
        num = nij(self.n0, self.n1) * (np.sin(theta) - np.sin(theta) * nij(self.n0, self.n1)**2 - np.sin(theta)**2 - nij(self.n0, self.n1)**2)
        den = np.cos(theta) * (2 * np.sin(theta)**2 - nij(self.n0, self.n1)**2)

        return self.l10TM(theta) * num/den

    def dt12TM(self,theta):
        """Goos-Hanchen shifts at the waveguide/superstrate TM polarization """
        num = nij(self.n2, self.n1) * (np.sin(theta) - np.sin(theta) * nij(self.n2, self.n1)**2 - np.sin(theta)**2 - nij(self.n2, self.n1)**2)
        den = np.cos(theta) * (2 * np.sin(theta)**2 - nij(self.n2, self.n1)**2)

        return self.l12TM(theta) * num/den

    def number_of_reflections_TE(self, theta):
        """Number of reflections for:
        L    : length of guide
        d    : thickness of guide
        theta: angle of TIR in guide
        """

        return self.L / (2* self.d * np.tan(theta) + self.dt10TE(theta) + self.dt12TE(theta))


    def __str__(self):
        st = f"""
        n0 (substrate)                 = {self.n0}
        n1 (iow)                       = {self.n1}
        n1 (bulk)                      = {self.n2}
        d (thickness iow in nm)        = {self.d/nm:5.1f} nm
        L (Length    iow in cm)        = {self.L/cm:5.1f} cm
        lamda (laser wavelength in nm) = {self.lamda/nm:5.1f} nm
        theta10 (TIR angle n1-n0)      = {self.tc10 * rad_to_deg:5.1f} deg
        theta20 (TIR angle n1-n2)      = {self.tc12 * rad_to_deg:5.1f} deg
        """
        return st


def print_R(R):
    for m, r in R.items():
        print(f" mode = {m}, solutions = {r}")


def find_roots_TE(IOW, mmax=2, tmin=0, tmax=np.pi/2, points=1000, eps=1e-2):
    """Find the roots of the eigenvalue equations
    Returns a dictionary of PRoots, keyed by the propagation mode ms
    """
    def find_root(m):
        DD = []
        T = []
        for iow in IOW:
            for theta in np.linspace(tmin,tmax,points):
                eg = eigen(iow, theta, m, 'TE')
                if abs(eg.eigen())<eps:
                    DD.append(iow.d)
                    T.append(theta)
                    break
        return PRoots(np.array(DD), np.array(T))
    R= {}
    for m in range(mmax):
        R[m] = find_root(m)

    return R


def tc(nl, nh):
    """Computes the angle of total internal reflection between a medium with high refraction index nh
    and a medium with low refraction index nl

    """
    return np.arcsin(nl/nh)


def nij(nl, nh):
    """ratio between low refraction index (nl) and high refraction index (nh)"""
    return nl/nh


def de(lamda, nl , nh, theta):
    """Distance of penetration from medium nh into medium nl


    $d = \left(\frac{\lambda}{4\pi n_h}\right)
        \frac{1}{\sqrt{\sin(\theta)^2 - (\frac{n_l}{n_h})^2 }}$

    """
    num = lamda/(4 * np.pi * nh)
    den =  np.sqrt(np.sin(theta)**2 - nij(nl, nh)**2)
    return num/den


def tTE2(nl,nh,theta):
    """|t|^2 (TE) = (Ie/Ii)TE
    TE is the TE polarization
    (Ie/Ii) is the transmitted interfacil intensity
    per unit of incident intensity

    """
    return nij(nl, nh) * 4 * np.cos(theta) /(1 - nij(nl, nh)**2)


def tTM2(nl,nh,theta):
    """|t|^2 (TM) = (Ie/Ii)TM
    TM is the TM polarization

    """
    num = 2 * np.sin(theta)**2 - nij(nl, nh)**2
    den = (1 + nij(nl, nh)**2) * np.sin(theta)**2 - nij(nl, nh)**2
    return tTE2(nl,nh,theta) * num/den


def ie(z,lamda,nl,nh,theta,mode='TM'):
    """ie = (Ie/Ii)
    Including exponential attenuation

    """
    if mode == "TM":
        return tTM2(nl,nh,theta) * np.exp(-z/de(lamda,nl,nh,theta))
    else:
        return tTE2(nl,nh,theta) * np.exp(-z/de(lamda,nl,nh,theta))


def le(lamda, nl, nh,theta, mode='TE'):
    """Evanescent path length in nl medium -->
    ie *d --> (Ie/Ii) de

    """

    if mode == "TM":
        return tTM2(nl,nh,theta) * de(lamda,nl,nh,theta)
    else:
        return tTE2(nl,nh,theta) * de(lamda,nl,nh,theta)


def dtTE(lamda, nl, nh, theta):
    """Goos-Hanchen shifts at the waveguide/substrate interfaces TE  """

    return le(lamda, nl, nh,theta, mode='TE') * (np.cos(theta) * np.sin(theta)) / nij(nl, nh)


def dtTM(lamda, nl, nh, theta):
    """Goos-Hanchen shifts at the waveguide/superstrate TM  """
    num = nij(nl, nh) * (np.sin(theta) - np.sin(theta) * nij(nl, nh)**2 - np.sin(theta)**2 - nij(nl, nh)**2)
    den = np.cos(theta) * (2 * np.sin(theta)**2 - nij(nl, nh)**2)

    return le(lamda, nl, nh,theta, mode='TM') * num/den


def dt10TE(theta):
    """Goos-Hanchen shifts at the waveguide/substrate interfaces TE  """
    return l10TE(theta) * (np.cos(theta) * np.sin(theta)) / nij(n0, n1)


def dt12TE(theta):
    """Goos-Hanchen shifts at the waveguide/superstrate TE  """
    return l12TE(theta) * (np.cos(theta) * np.sin(theta)) / nij(n2, n1)


def dt10TM(theta):
    """Goos-Hanchen shifts at the waveguide/superstrate TM  """
    num = nij(n0, n1) * (np.sin(theta) - np.sin(theta) * nij(n0, n1)**2 - np.sin(theta)**2 - nij(n0, n1)**2)
    den = np.cos(theta) * (2 * np.sin(theta)**2 - nij(n0, n1)**2)

    return l10TM(theta) * num/den


def dt12TM(theta):
    """Goos-Hanchen shifts at the waveguide/superstrate TM polarization """
    num = nij(n2, n1) * (np.sin(theta) - np.sin(theta) * nij(n2, n1)**2 - np.sin(theta)**2 - nij(n2, n1)**2)
    den = np.cos(theta) * (2 * np.sin(theta)**2 - nij(n2, n1)**2)

    return l12TM(theta) * num/den


def eigen(iow, theta, m, mode='TE'):
    """
    Eigenvalue equation for propagation modes m.
    """
    def kite(lamda, theta, nl, nh):
        return (2 * np.pi/lamda) * np.sqrt(np.abs(nh**2*np.sin(theta)**2 - nl**2))

    def kitm(lamda, theta, nl, nh):
        return kite(lamda, theta, nl, nh) / nl**2

    if mode == 'TE':
        ki = kite
    else:
        ki = kitm

    t1 = ki(iow.lamda, theta, iow.n1, iow.n1) * iow.d
    t2 = np.arctan2(ki(iow.lamda, theta, iow.n2, iow.n1),ki(iow.lamda, theta, iow.n1, iow.n1))
    t3 = np.arctan2(ki(iow.lamda, theta, iow.n0, iow.n1),ki(iow.lamda, theta, iow.n1, iow.n1))
    t4 = m * np.pi
    return Eigen(t1, t2, t3, t4)


def number_of_reflections_TE(iow, theta):
    """Number of reflections for:
    L    : length of guide
    d    : thickness of guide
    theta: angle of TIR in guide
    """

    return iow.L / (2* iow.d * np.tan(theta) + dtTE(iow.lamda, iow.n0, iow.n1, theta) + dtTE(iow.lamda, iow.n2, iow.n1, theta))
