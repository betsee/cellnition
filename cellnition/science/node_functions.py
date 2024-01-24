#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module contains different functions that can be applied at the nodes when constructing an analytical
model. These functions are intended to be used with symbolic computing (sympy).
'''
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from scipy.optimize import minimize, fsolve
from scipy.signal import square
import networkx as nx
from networkx import DiGraph
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
import pygraphviz as pgv


def f_acti_s(cc, beta, nn):
    '''

    '''
    return ((cc * beta) ** nn) / (1 + (cc * beta) ** nn)


def f_inhi_s(cc, beta, nn):
    '''

    '''
    return 1 / (1 + (cc * beta) ** nn)


def f_neut_s(cc, kk, nn):
    '''
    Calculates a "neutral" edge interaction, where
    there is neither an activation nor inhibition response.
    '''
    return 1