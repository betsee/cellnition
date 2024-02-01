#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module contains different functions that can be applied as the influence of one node
over another node (node-node interaction) when constructing an analytical
model. These functions are intended to be used with symbolic computing (sympy).
'''
from numpy import ndarray
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from sympy import MatrixSymbol


def f_hill_s(i, j, pp: MatrixSymbol, nn: MatrixSymbol, beta: MatrixSymbol):
    '''

    '''
    return 1/(1 + (beta[j,i] * pp[j,i]) ** -nn[j,i])


def f_logi_s(i, j, pp: MatrixSymbol, kk: MatrixSymbol, mu: MatrixSymbol):
    '''

    '''
    return 1 / (1 + sp.exp(-kk[j,i]*(pp[j,i] - mu[j,i])))

def f_acti_hill_s(cc: Symbol|Indexed, beta: Symbol|Indexed, nn: Symbol|Indexed):
    '''
    Activator function based on a Hill function.
    The entity, cc, will be activating another node.

    Parameters
    ----------
    cc : float|ndarray|list
        Concentration or set of concentrations at which
        to compute the function.

    beta: float
        The network Hill coefficient, which is equal to the
        maximum rate of production of cc divided by the
        decay of cc multiplied by the standard Hill coefficient:
        (beta = r_max/(d_max*K_edge)).

    nn : float
        The Hill exponent.

    '''
    return ((cc * beta) ** nn) / (1 + (cc * beta) ** nn)

def f_inhi_hill_s(cc: Symbol|Indexed, beta: Symbol|Indexed, nn: Symbol|Indexed):
    '''
    Inhibitor function based on a Hill function.
    The entity, cc, will be inhibiting another node.

    Parameters
    ----------
    cc : float|ndarray|list
        Concentration or set of concentrations at which
        to compute the function.

    beta: float
        The network Hill coefficient, which is equal to the
        maximum rate of production of cc divided by the
        decay of cc multiplied by the standard Hill coefficient:
        (beta = r_max/(d_max*K_edge)).

    nn : float
        The Hill exponent.

    '''
    return 1 / (1 + (cc * beta) ** nn)

def f_neut_s(cc: Symbol|Indexed, kk: Symbol|Indexed, nn: Symbol|Indexed):
    '''
    Calculates a "neutral" edge interaction, where
    there is neither an activation nor inhibition response.
    '''
    return 1

def f_acti_logi_s(cc: Symbol|Indexed, co: Symbol|Indexed, k: Symbol|Indexed):
    '''
    Activator function based on a logistic function.
    The entity, cc, will be activating another node.
    This function can only be used in symbolic Sympy
    equations.

    Parameters
    ----------
    cc : float|ndarray|list
        Concentration or set of concentrations at which
        to compute the function.

    co: float
        The centre of the sigmoidal logistic curve.

    k : float
        The coupling strength/rise function. Here k>0 to
        achieve an activator response.

    '''
    return 1/(1 + sp.exp(-k*(cc - co)))

def f_inhi_logi_s(cc: Symbol|Indexed, co: Symbol|Indexed, k: Symbol|Indexed):
    '''
    Activator function based on a logistic function.
    The entity, cc, will be activating another node.
    This function can only be used in symbolic Sympy
    equations.

    Parameters
    ----------
    cc : float|ndarray|list
        Concentration or set of concentrations at which
        to compute the function.

    co: float
        The centre of the sigmoidal logistic curve.

    k : float
        The coupling strength/rise function. Here k>0 to
        achieve an inhibition response.

    '''
    return 1/(1 + sp.exp(k*(cc - co)))