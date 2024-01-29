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
    attractor = 'Stable Attractor'
    repellor = 'Stable Repellor'
    limit_cycle = 'Stable Limit Cycle'
    cycle = 'Limit Cycle'
    saddle = 'Saddle Attractor'
    undetermined = 'Undetermined'

class InterFuncType(Enum):
    logistic = 'Logistic'
    hill = 'Hill'

class CouplingType(Enum):
    additive = 'additive'
    multiplicative = 'multiplicative'
    mixed = 'mixed'
    specified = 'specified'
