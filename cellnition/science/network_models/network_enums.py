#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines enumerations used in Cellnition.
'''


from enum import Enum

class EdgeType(Enum):
    '''

    '''
    A = 'Activator'
    I = 'Inhibitor'
    N = 'Neutral'
    As = 'Multiplicative Activation'
    Is = 'Multiplicative Inhibition'


class NodeType(Enum):
    '''

    '''
    gene = 'Gene'
    signal = 'Signal'
    process = 'Process'
    sensor = 'Sensor'
    effector = 'Effector'
    core = 'Hub Core'
    factor = 'Factor'
    cycle = 'Cycle'

class GraphType(Enum):
    '''

    '''
    scale_free = 'Scale Free'
    random = 'Random'
    user = 'User Defined'

class PType(Enum):
    '''

    '''
    in_deg = 'in degree'
    out_deg = 'out degree'
    ave_deg = 'ave degree'
    rand = 'random'

class EquilibriumType(Enum):
    '''

    '''
    attractor = 0
    attractor_limit_cycle = 1
    limit_cycle = 2
    saddle = 3
    repellor = 4
    repellor_limit_cycle = 5
    undetermined = 6
    hidden = 7 # corresponds to a hidden attractor

class InterFuncType(Enum):
    '''

    '''
    logistic = 'Logistic'
    hill = 'Hill'

class CouplingType(Enum):
    '''

    '''
    additive = 'additive'
    multiplicative = 'multiplicative'
    mix1 = 'mix1' # activators "OR", inhibitors "AND"
    specified = 'specified'
    mix2 = 'mix2' # Activators "AND", inhibitors "OR"
