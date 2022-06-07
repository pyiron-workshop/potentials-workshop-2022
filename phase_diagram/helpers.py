import pandas as pd
import os
import numpy as np
from calphy.integrators import kb
from scipy.optimize import fsolve

def create_structures(pr, comp, repetitions=4):
    """
    Create off stoichiometric structures
    
    Parameters
    ----------
    pr: pyiron project
        
    comp: required composition, float
    
    repetitions: int
        required super cell size
        
    """
    structure_fcc = pr.create.structure.ase.bulk('Al', cubic=True, a=4.135).repeat(repetitions)
    n_li = int(comp*len(structure_fcc))
    structure_fcc[np.random.permutation(len(structure_fcc))[:n_li]] = 'Li'
    
    structure_b32 = pr.create.structure.ase.read('AlLi_poscar', format='vasp')
    n_li = int((0.5-comp)*len(structure_b32))
    rinds = len(structure_b32)//2 + np.random.choice(range(len(structure_b32)//2), n_li, replace=False)
    structure_b32[rinds] = 'Al'
    return structure_fcc, structure_b32
    
    
def fe_at(p, temp, threshold=1E-1):
    """
    Get the free energy at a given temperature
    
    Parameters
    ----------
    p: pyiron Job
        Pyiron job with calculated free energy and temperature
        
    temp: float
        Required temperature
        
    threshold: optional, default 1E-1
        Minimum difference needed between required temperature and temperature found in pyiron job
        
    Returns
    -------
    float: free energy value at required temperature
    """
    arg = np.argsort(np.abs(p.output.temperature-temp))[0]
    th = np.abs(p.output.temperature-temp)[arg] 
    if th > threshold:
        raise ValueError("not a close match, threshold %f"%th)
    return p.output.energy_free[arg]

def normalise_fe(fe_arr, conc_arr):
    """
    Get the enthalpy of mixing by fitting and subtracting a straight line connecting the end points.
    
    Parameters
    ----------
    fe_arr: list of floats
        array of free energy values as function of composition
        
    conc_arr: list of floats
        array of composition values
    
    Returns
    -------
    norm: list of floats
        normalised free energy
    
    m: float
        slope of the fitted line
    
    c: float
        intercept of the fitted line
    """
    m = (fe_arr[-1]-fe_arr[0])/(conc_arr[-1]-conc_arr[0])
    c = fe_arr[-1]-m*(conc_arr[-1]-conc_arr[0])
    norm = fe_arr-(m*conc_arr+c)
    return norm, m, c

def find_common_tangent(fe1, fe2, guess_range):
    """
    Do a common tangent construction between two free energy curves.
    
    Parameters
    ----------
    fe1: numpy array
        first free energy curve
    
    fe2: numpy array
        second free energy curve
    
    guess_range: list of floats length 2
        The guess range to find end points of the common tangent
    
    Returns
    -------
    res: list of floats length 2
        The end points of the common tangent
    """
    def _ct(x, p1, p2):
        p1der = np.polyder(p1)
        p2der = np.polyder(p2)
        term1 = np.polyval(p1der, x[0])-np.polyval(p2der, x[1])
        term2 = (np.polyval(p1, x[0]) - np.polyval(p2, x[1]))/(x[0]-x[1]) - np.polyval(p1der, x[0])
        return [term1, term2]
    res = fsolve(_ct, guess_range, args=(fe1, fe2))
    return res