#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines enumerations used in Cellnition.
'''


from enum import Enum

class EdgeType(Enum):
    A = 'Activator'
    I = 'Inhibitor'
    N = 'Neutral'
    As = 'Multiplicative Activation'
    Is = 'Multiplicative Inhibition'


class NodeType(Enum):
    gene = 'Gene'
    signal = 'Signal'
    process = 'Process'
    sensor = 'Sensor'
    effector = 'Effector'
    root = 'Root Hub'
    factor = 'Factor'
    cycle = 'Cycle'

class GraphType(Enum):
    scale_free = 'Scale Free'
    random = 'Random'
    user = 'User Defined'

class EquilibriumType(Enum):
    attractor = 'Attractor'
    repellor = 'Repellor'
    attractor_limit_cycle = 'Attractor Limit Cycle'
    repellor_limit_cycle = 'Repellor Limit Cycle'
    limit_cycle = 'Limit Cycle'
    saddle = 'Saddle'
    undetermined = 'Undetermined'

class InterFuncType(Enum):
    logistic = 'Logistic'
    hill = 'Hill'

class CouplingType(Enum):
    additive = 'additive'
    multiplicative = 'multiplicative'
    mixed = 'mixed'
    specified = 'specified'
