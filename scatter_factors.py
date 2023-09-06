#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 6 12:05:10 2023

@author: gjadick

A script for retrieving complex index of refraction values
using the Photon Anomalous Scattering Factors from EPDL97 
(https://www-nds.iaea.org/epdl97/).


#####################################################################
### EXAMPLE USAGE
#####################################################################

To get delta and beta factors for water as a function of energy:

```
energies = np.arange(1., 140., 1.)  # array of energy in keV
material = 'H(88.8)O(11.2)'         # water composition, using 'name' elem_format
density = 1.0                       # material density in g/cm^3

delta, beta = get_delta_beta_mix(matcomp, energies, density, elem_format='name')
```


#####################################################################
### NOTES
#####################################################################

The package `periodictable` includes a method for getting these
values, but this uses data from Henke et al., which is considered less
accurate at energies above 30 keV [Jacobsen]. 

Each element's data is saved in a separate file. As explained by IAEA, 
the data format for each file is :

    The first line defines,
    1)  The Atomic Number of the Element (Z = 1 for hydrogen)
    2)  The Number of Tabulated Energy Points (361)
    3)  The Atomic Weight of the Naturally Occurring Element (1.008)
    4)  The STP Density in grams/cc (8.988e-5)
    5)  Definition of the Element in text (1-H -Nat)  
     
    The Second Line contains titles for each Column
    1)  MeV - the units of Energy
    2)  F1-Total - Total f1 (sum of ionization and excitation)
    3)  Z+F1 - In this form it is easier to see FF+F1 approaching zero at low energy.   
    4)  F1-Ionize - Contribution of Ionization to f1
    5)  F1-Excite - Contribution of Excitation to f1
    6)  F2-Total - Total f2 (sum of ionization and excitation)
    7)  F2-Ionize - Contribution of Ionization to f2
    8)  F2-Excite - Contribution of Excitation to f2
    9)  Coherent (barns) - Coherent Cross Section defined by integrating the above equation.
    
"""

import os
import re
import numpy as np

# try:
#     rootpath = os.path.dirname(__file__)
# except:
#     rootpath = ''
rootpath = 'input/scatter_data'  # !!! might need to change this !!!


### CONSTANTS
r_e = 2.8179403262e-15        # classical electron radius, m
N_A = 6.02214076e+23          # Avogadro's number, num/mol
pi = np.pi
h  = 6.62607015e-34           # Planck constant, J/Hz
c = 299792458.0               # speed of light, m/s
J_eV = 1.602176565e-19        # J per eV conversion
ELEM_FORMAT_DEFAULT = 'name'  # 'Z' or 'name'

elements =  ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si',\
    'P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',\
    'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh',\
    'Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',\
    'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re',\
    'Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th',\
    'Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm']


def get_wavelen(energy):
    # energy in keV -> returns wavelength in m
    return 1e-3*h*c/(energy*J_eV)

def get_filename(elem, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Get the filename with data for a given element.

    INPUTS:
    elem - atomic number (int, 'Z' format)
           or short element ID (str)
    elem_format - 'Z' or 'name' (optional)

    OUTPUTS:
    filename - the path to the file
    """
    assert elem_format=='Z' or elem_format=='name' 
    if elem_format=='Z':
        Z = elem
    else:
        Z = elements.index(elem)+1
    filename =  os.path.join(rootpath, f'za{int(Z):03}000.txt')
    return filename


def to_float(s):
    '''
    Convert an EPDL97-formatted number to a float.
    Their scientific notation format for "X * 10^-Y" 
    is "X-Y", or for "X * 10+Y" it is "X+Y", 
    neither of which can convert to float using 
    Python type casting alone.
    
    Perhaps this function is excessively convoluted;
    I welcome simpler solutions.
    '''
    try:    # attempt type casting
        result = float(s)
    except: # convert string
        # check for negative val
        sign = 1
        if s[0]=='-':
            sign = -1 
            s = s[1:]
        # get power
        if len(s[1:].split('+'))==2:
            split_char = '+'
            pow_sign = 1.0
        elif len(s[1:].split('-'))==2:
            split_char = '-'
            pow_sign = -1.0
        else:
            print('FLOAT CONVERSION ERROR', s)
            return None
        # calc float
        A = float(s.split(split_char)[0])
        pow = pow_sign * float(s.split(split_char)[1])
        result = sign * A * 10**pow
        
    return result


def get_amass(elem, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Finds the atomic mass of an element in g/mole.

    INPUTS:
    elem - atomic number (int, 'Z' format)
           or short element ID (str)
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    amass - double specifiying atomic mass in grams per mole
    """
    f = open(get_filename(elem, elem_format))
    amass = to_float(f.readline().split()[2])
    f.close()
    return amass


def get_f1_f2(elem, energy, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Get energy-dependent anomalous scatter factors for a single element.

    INPUTS:
    elem - atomic number (int, 'Z' format)
           or short element ID (str)
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    f1, f2 - arrays of scatter factors as function of energy
    """
    n = 10  # number of characters per column in data file
    f = open(get_filename(elem, elem_format))
    data = np.array([[line[i:i+n] for i in range(0, len(line)-1, n)] for line in f.readlines()[2:]]).T
    f.close()

    # get table of keV, f1, f2
    keV_0 = 1000 * np.array([to_float(x) for x in data[0]])
    f1_0 = np.array([to_float(x) for x in data[2]])
    f2_0 = np.array([to_float(x) for x in data[5]])
    
    # linear interp to energy vals
    f1 = np.interp(energy, keV_0, f1_0)
    f2 = np.interp(energy, keV_0, f2_0)
    
    return f1, f2


def get_delta_beta(elem, energy, density, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Get energy-dependent phase (delta) and absorption (beta) parameters.

    INPUTS:
    elem - atomic number (int, 'Z' format)
           or short element ID (str)
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    density - element density in g/cm^3 (float)
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    delta, beta - arrays of scatter factors as function of energy
    """
    # get some scaling values
    A = get_amass(elem, elem_format) # atomic mass
    n_a = (10**6)*density*N_A/A               # number density
    alpha = n_a*r_e/(2*pi)            # scale factor
    wavelen = get_wavelen(energy)    # wavelengths [m]

    # get the scatter factors
    f1, f2 = get_f1_f2(elem, energy, elem_format)
    delta = alpha * wavelen**2 * f1
    beta = alpha * wavelen**2 * f2

    return delta, beta


### HANDLING MIXTURES - USE STOICHIOMETRIC WEIGHTING
### Weighted average of atomic masses
### Implemented following Jacobsen ch 3, eq'n 3.101-103


def parse_matcomp(name):
    """
    Converts string of material composition to list of material names 
    and a list of their weightings.
    
    INPUTS:
    name - matcomp string, e.g. for water 'H(88.8)O(11.2)'

    OUTPUTS:
    matnames - list of individual element names, e.g. ['H','O']
    weights - list of corresponding weights, e.g. [88.8, 11.2]
    """
    
    matnames = []
    weights = []

    ii = 0
    sub = name
    lp = sub.find('(')
    rp = sub.find(')')
    while lp != -1:
        matnames.append(sub[:lp])
        weights.append(float(sub[(lp+1):rp]))
        ii = rp+1
        sub = sub[ii:]
        lp = sub.find('(')
        rp = sub.find(')')
        
    # normalize weights
    weights = np.array(weights)
    weights = weights/np.sum(weights)
    
    return matnames, weights 
    

def get_f1_f2_mix(matcomp, energy, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Get energy-dependent anomalous scatter factors for a mixture
    of elements with given weighting factors.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    f1, f2 - arrays of scatter factors as function of energy
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)
    N_energy = np.array(energy).size

    # get individual element f1, f2
    f1_elems_weighted = np.zeros([N_elems, N_energy])
    f2_elems_weighted = np.zeros([N_elems, N_energy])
    for i in range(N_elems):
        elem = matnames[i]
        s_i = weights[i]
        f1, f2 = get_f1_f2(elem, energy, elem_format)
        f1_elems_weighted[i] = s_i * f1
        f2_elems_weighted[i] = s_i * f2

    # compute weighted averages as func of energy
    f1_avg = np.sum(f1_elems_weighted, axis=0)
    f2_avg = np.sum(f2_elems_weighted, axis=0)

    return f1_avg, f2_avg


def get_amass_mix(matcomp, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Finds the average atomic mass of a mixture.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    amass - double specifiying avg atomic mass in grams per mole
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)

    amass_elems_weighted = np.zeros(N_elems)
    for i in range(N_elems):
        elem = matnames[i]
        s_i = weights[i]
        amass = get_amass(elem, elem_format)
        amass_elems_weighted[i] = s_i * amass

    # compute weighted average
    amass = np.sum(amass_elems_weighted)

    return amass
    


def get_delta_beta_mix(matcomp, energy, density, elem_format=ELEM_FORMAT_DEFAULT):
    """
    Get energy-dependent phase (delta) and absorption (beta) parameters
    for a mixture of elements with given weighting factors.

    INPUTS:
    matcomp - material composition and weights (str)
              e.g. for water 'H(88.8)O(11.2)'
    energy - energies at which to evaluate factors 
             (number or list/array of numbers)
    density - element density in g/cm^3 (float)
    elem_format - 'Z' or 'name' (optional)
    
    OUTPUTS:
    delta, beta - arrays of scatter factors as function of energy
    """
    matnames, weights = parse_matcomp(matcomp)
    N_elems = len(matnames)
    N_energy = np.array(energy).size

    # check for mixture or single
    if N_elems == 1: # single element
        elem = matnames[0]
        A = get_amass(elem, elem_format) # atomic mass
        f1, f2 = get_f1_f2(elem, energy, elem_format)
    else:
        A = get_amass_mix(matcomp, elem_format)
        f1, f2 = get_f1_f2_mix(matcomp, energy, elem_format)

    # some scale factors
    n_a = (10**6)*density*N_A/A      # number density 
    alpha = n_a*r_e/(2*pi)   # scale factor 
    wavelen = get_wavelen(energy)    # wavelengths [m]

    # compute the scatter factors
    delta = alpha * wavelen**2 * f1
    beta = alpha * wavelen**2 * f2

    return delta, beta






