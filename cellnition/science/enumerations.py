#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines
'''


from enum import Enum

class EdgeType(Enum):
    A = 'Activator'
    I = 'Inhibitor'
    N = 'Normal'
    As = 'Signal Activation'
    Is = 'Signal Inhibition'


class NodeType(Enum):
    gene = 'Gene'
    signal = 'Signal'
    process = 'Process'
    sensor = 'Sensor'
    effector = 'Effector'
    root = 'Root Hub'
    path = 'Path'
    factor = 'Factor'

class GraphType(Enum):
    scale_free = 'Scale Free'
    random = 'Random'

class EquilibriumType(Enum):
    attractor = 'Stable Attractor'
    repellor = 'Stable Repellor'
    limit_cycle = 'Stable Limit Cycle'
    cycle = 'Limit Cycle'
    saddle = 'Saddle Attractor'
    undetermined = 'Undetermined'