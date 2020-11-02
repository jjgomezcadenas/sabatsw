import numpy as np
import pandas as pd
import os, sys

from  invisible_cities.core.system_of_units import *
import matplotlib.pyplot as plt

from pandas import DataFrame, Series
from matplotlib import cm as cmp

from sabat.sbt_types import  photon, molecule, GM, us, ucm, ucm2, ucm3, gp
from sabat.sbt_types import  umm, umum, mum2, mW
from sabat.sbt_iow import  eigen, ie

rad_to_deg = 180/np.pi

def plot_depth_of_penetration(iow, figsize=(10,6), eps=1e-2):

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)

    TH10 = np.linspace(iow.tc10 + eps, np.pi/2, 100)
    TH12 = np.linspace(iow.tc12 + eps, np.pi/2, 100)

    plt.plot(TH10 * rad_to_deg, iow.d10(TH10)/nm, lw=2, label=r'$d_{1,0}$')
    plt.axvline(x=iow.tc10 * rad_to_deg, ymin=0, ymax=160, color='r', linestyle='dashed')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'd (nm)')
    plt.legend()

    ax = plt.subplot(1,2,2)
    plt.plot(TH12 * rad_to_deg, iow.d12(TH12)/nm, lw=2, label=r'$d_{1,2}$')
    plt.axvline(x=iow.tc12 * rad_to_deg, ymin=0, ymax=160, color='r', linestyle='dashed')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'd (nm)')
    plt.legend()

    plt.show()


def plot_Ie_Ii(iow, figsize=(10,6),eps=1e-2):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)

    TH10 = np.linspace(iow.tc10 + eps, np.pi/2, 100)
    TH12 = np.linspace(iow.tc12 + eps, np.pi/2, 100)

    plt.plot(TH10 * rad_to_deg, iow.Ie_Ii_TE_10(TH10), lw=2, label=r'$(I_e/I_i)_{1,0}^{TE}$')
    plt.plot(TH10 * rad_to_deg, iow.Ie_Ii_TM_10(TH10), lw=2, label=r'$(I_e/I_i)_{1,0}^{TM}$')

    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'per unit')
    plt.legend()

    ax = plt.subplot(1,2,2)
    plt.plot(TH12 * rad_to_deg, iow.Ie_Ii_TE_12(TH12), lw=2, label=r'$(I_e/I_i)_{1,2}^{TE}$')
    plt.plot(TH12 * rad_to_deg, iow.Ie_Ii_TM_12(TH12), lw=2, label=r'$(I_e/I_i)_{1,2}^{TM}$')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'$(I_e/I_i)$')
    plt.legend()

    plt.show()


def plot_evanescent_path_length(iow, figsize=(10,6), eps=1e-2):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)

    TH10 = np.linspace(iow.tc10 + eps, np.pi/2, 100)
    TH12 = np.linspace(iow.tc12 + eps, np.pi/2, 100)

    plt.plot(TH10 * rad_to_deg, iow.l10TE(TH10)/nm, lw=2, label=r'$L_{1,0}^{TE}$')
    plt.plot(TH10 * rad_to_deg, iow.l10TM(TH10)/nm, lw=2, label=r'$L_{1,0}^{TM}$')

    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'per unit')
    plt.legend()

    ax = plt.subplot(1,2,2)
    plt.plot(TH12 * rad_to_deg, iow.l12TE(TH12)/nm, lw=2, label=r'$L_{1,2}^{TE}$')
    plt.plot(TH12 * rad_to_deg, iow.l12TM(TH12)/nm, lw=2, label=r'$L_{1,2}^{TM}$')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'L (nm)')
    plt.legend()

    plt.show()


def plot_GH_shifts(iow, figsize=(10,6), eps=1e-2):
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1,2,1)

    TH10 = np.linspace(iow.tc10 + eps, np.pi/2, 100)
    TH12 = np.linspace(iow.tc12 + eps, np.pi/2, 100)

    plt.plot(TH10 * rad_to_deg, iow.dt10TE(TH10)/nm, lw=2, label=r'$\Delta_{1,0}^{TE}$')
    plt.plot(TH10 * rad_to_deg, iow.dt10TM(TH10)/nm, lw=2, label=r'$\Delta_{1,0}^{TM}$')

    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'$\Delta$')
    plt.legend()

    ax = plt.subplot(1,2,2)
    plt.plot(TH12 * rad_to_deg, iow.dt12TE(TH12)/nm,  lw=2, label=r'$\Delta_{1,2}^{TE}$')
    plt.plot(TH12 * rad_to_deg, iow.dt12TM(TH12)/nm,  lw=2, label=r'$\Delta_{1,2}^{TM}$')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'$\Delta$ (nm)')
    plt.legend()

    plt.show()


def plot_eigenvalues(IIOW, mode='TE', mmax=4, figsize=(14,8), eps=1e-2):

    fig = plt.figure(figsize=figsize)

    M  = np.arange(0,mmax)

    for i, iow in enumerate(IIOW):
        #print(iow)
        ax = plt.subplot(2,2,i+1)
        TH10 = np.linspace(iow.tc10 + eps, np.pi/2, 100)
        TH12 = np.linspace(iow.tc12 + eps, np.pi/2, 100)

        for m in M:
            eg = eigen(iow, TH12, m, mode)
            plt.plot(TH12 * rad_to_deg, eg.eigen(), lw=2, label=f'd = {iow.d/nm:5.1f} nm m = {m}')
            plt.legend()
            plt.axhline(y=0, color='k', linestyle='dashed')
    plt.tight_layout()
    plt.show()


def plot_roots_TE(R, m, figsize=(14,8)):
    fig = plt.figure(figsize=figsize)
    TH = np.array([pr.theta for pr in R.values()])
    D = np.array([pr.d for pr in R.values()])
    plt.plot(TH * rad_to_deg, D/nm, 'bo')
    plt.xlabel(r'$\theta$ (deg)')
    plt.ylabel(r'd (nm)')
    plt.show()


def plot_evanescent_field(R, iow, m, figsize=(8,8)):
    fig = plt.figure(figsize=figsize)
    PR = R[m]
    for i, r in enumerate(PR.d):
        theta = PR.theta[i]
        ax = plt.subplot(2,2,i+1)
        Z  = np.linspace(0, 200 *nm, 100)
        I = ie(Z,iow.lamda, iow.n0, iow.n1, theta,mode='TE')
        plt.plot(Z/nm, I, lw=2, label=f'r = {r/nm:5.1f}, theta={theta * rad_to_deg:5.1f}')
        plt.legend()
    plt.tight_layout()
    plt.show()
