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

class NodeType(Enum):
    gene = 'Gene Product'
    process = 'Process'
    sensor = 'Sensor'
    effector = 'Effector'