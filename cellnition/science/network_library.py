#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines several types of study networks.
'''
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from scipy.optimize import minimize, fsolve
from scipy.signal import square
import networkx as nx
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.enumerations import EdgeType, GraphType, NodeType
from cellnition.science.stability import Solution
import pygraphviz as pgv
import pyvista as pv

class LibNet(object, metaclass=ABCMeta):
    '''

    '''
    def __init__(self):
        '''

        '''

        pass

class BinodeNetAdd(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeNet_AddInteract'
        self.short_name = '2Net_Add'

        self.N_nodes = 2
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('H0', 'H0'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A,
                      EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is
                      ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True
class QuadStateNetAdd(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'QuadStateNet_AddInteract'
        self.short_name = 'QuadState_Add'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,
                      ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True
class QuadStateNetMult(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'QuadStateNet_MultInteract'
        self.short_name = 'QuadState_Mult'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                      ]

        self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                           EdgeType.A, EdgeType.A, EdgeType.A,
                           EdgeType.As, EdgeType.As, EdgeType.As,
                           ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = False
class FullQuadStateNetAdd(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'FullQuadStateNet_AddInteract'
        self.short_name = 'FullQuadState_Add'

        # CASE TYPE QUADSTABLE with sensors and auxillary nodes in scale-free configuration:
        # Core is triangle loop with all auto-nodes edges:
        self.N_nodes = 14
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                 ('H2', 'G0'), ('H2', 'G1'), ('H2', 'G2'), ('H2', 'G3'), ('H2', 'G4'),
                 ('H0', 'G5'), ('H0', 'G6'), ('H0', 'G7'),
                 ('H1', 'G8')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,
                      EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.I, EdgeType.A,
                      EdgeType.A
                      ]

        # non-homogeneous K-vects to see more effect in satelite nodes:
        self.K_vect = [0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.25, 0.5, 1.0, 1.5, 0.5,
                   0.5, 0.5, 1.5,
                   0.5
                   ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True