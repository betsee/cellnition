#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines several types of study networks.
'''
from abc import ABCMeta, abstractmethod
import numpy as np
from cellnition.science.network_enums import EdgeType, GraphType, NodeType


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
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeNet_AddInteract'
        self.short_name = 'BinodeNet_Add'

        self.N_nodes = 2
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('H0', 'H0'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.I,
                      EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is
                      ]

        # Generates set of 4 nicely spaced attractors:
        K_vect = [0.57857143, 1.43142857, 1.14714286, 0.57857143, 0.5, 0.5]
        self.B_vect = (1/np.asarray(K_vect)).tolist()
        self.d_vect = 1.0
        self.n_vect = 3.0
        self.cmax = 4.0

        K_vect_alt1 = [1.43142857, 0.57857143, 0.57857143, 1.14714286, 0.5, 0.5]
        self.B_vect_alt1 = (1 / np.asarray(K_vect_alt1)).tolist()
        self.d_vect_alt1 = 1.0
        self.n_vect_alt1 = 3.0
        self.cmax_alt1 = 4.0

        K_vect_alt2 = 0.5
        self.B_vect_alt2 = (1/K_vect_alt2)
        self.d_vect_alt2 = 1.0
        self.n_vect_alt2 = 3.0
        self.cmax_alt2 = 4.0

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BasicBinodeNet(LibNet):

    def __init__(self):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeNet_Simple'
        self.short_name = 'SimpleBinodeNet'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 # ('H0', 'H0'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.I,
                      # EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeNetAdd(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeNet_AddInteract'
        self.short_name = 'TrinodeNet_Add'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True


        # Penta-state, well spaced in all vectors on triangle (I, I, I)
        K_vect = [0.45, 0.45, 0.1 , 0.45, 0.45, 0.1, 0.5, 0.5, 0.5]
        self.B_vect = (1 / np.asarray(K_vect)).tolist()
        self.d_vect =  1.0
        self.n_vect = 3.0
        self.cmax = 4.0

        # Penta-state, well spaced in all vectors on triangle (I, I, I)
        K_vect_alt1 = [0.45, 0.1, 0.45, 0.45, 0.45, 0.45, 0.5, 0.5, 0.5]
        self.B_vect_alt1 = (1 / np.asarray(K_vect_alt1)).tolist()
        self.d_vect_alt1 =  1.0
        self.n_vect_alt1 = 3.0
        self.cmax_alt1 = 4.0

        K_vect_alt2 = 0.5
        self.B_vect_alt2 = 1/K_vect_alt2
        self.d_vect_alt2 = 1.0
        self.n_vect_alt2 = 3.0
        self.cmax_alt2 = 4.0

class BasicTrinodeNet(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeNet_Simple'
        self.short_name = 'SimpleTrinodeNet'

        self.N_nodes = 3
        self.edges = [('C0', 'C1'), ('C1', 'C2'), ('C2', 'C0'),
                 # ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                      # EdgeType.A, EdgeType.A, EdgeType.A,
                      # EdgeType.Is, EdgeType.Is, EdgeType.Is,
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        self.node_type_dict = None

        self.add_interactions = True


        # Penta-state, well spaced in all vectors on triangle (I, I, I)
        K_vect = [0.45, 0.45, 0.1 , 0.45, 0.45, 0.1, 0.5, 0.5, 0.5]
        self.B_vect = (1 / np.asarray(K_vect)).tolist()
        self.d_vect =  1.0
        self.n_vect = 3.0
        self.cmax = 4.0

        # Penta-state, well spaced in all vectors on triangle (I, I, I)
        K_vect_alt1 = [0.45, 0.1, 0.45, 0.45, 0.45, 0.45, 0.5, 0.5, 0.5]
        self.B_vect_alt1 = (1 / np.asarray(K_vect_alt1)).tolist()
        self.d_vect_alt1 =  1.0
        self.n_vect_alt1 = 3.0
        self.cmax_alt1 = 4.0

        K_vect_alt2 = 0.5
        self.B_vect_alt2 = 1/K_vect_alt2
        self.d_vect_alt2 = 1.0
        self.n_vect_alt2 = 3.0
        self.cmax_alt2 = 4.0

class PentanodeNetAdd(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'PentanodeNet_AddInteract'
        self.short_name = 'PentanodeNet_Add'

        self.N_nodes = 5
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'H4'), ('H4', 'H0'),
                 ('H0', 'H0'), ('H1', 'H1'), ('H2', 'H2'), ('H3', 'H3'), ('H4', 'H4'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3'), ('S4', 'H4')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,EdgeType.Is, EdgeType.Is
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

        # 12 state net -- good:
        K_vect = [0.05,  0.05,  0.5,  0.5,  0.5,  0.5,  0.5,  0.05,  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.B_vect = (1 / np.asarray(K_vect)).tolist()
        self.d_vect =  1.0
        self.n_vect = 3.0
        self.cmax = 4.0

        # 11 state net -- good:
        K_vect_alt1 = [0.05,  0.05,  0.05,  0.05,  0.05,  0.5,  0.5,  0.5,  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.B_vect_alt1 = (1 / np.asarray(K_vect_alt1)).tolist()
        self.d_vect_alt1 =  1.0
        self.n_vect_alt1 = 3.0
        self.cmax_alt1 = 4.0

        K_vect_alt2 = 0.5
        self.B_vect_alt2 = 1 / K_vect_alt2
        self.d_vect_alt2 = 1.0
        self.n_vect_alt2 = 3.0
        self.cmax_alt2 = 4.0

class TrinodeNetLoaded(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeNet_AddInteract_Loaded'
        self.short_name = 'TrinodeNet_Loaded'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H1', 'H0'), ('H2', 'H1'), ('H0', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                     ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                     ]

        self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                           EdgeType.I, EdgeType.I, EdgeType.I,
                           EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

        # OK hex-state
        K_vect = [0.02, 0.02, 0.02, 0.31, 0.31, 0.6 , 0.31, 0.6 , 0.6, 0.5, 0.5, 0.5]
        self.B_vect = (1 / np.asarray(K_vect)).tolist()
        self.d_vect =  1.0
        self.n_vect = 3.0
        self.cmax = 4.0

        # OK hex-state
        K_vect_alt1 = [0.02, 0.02, 0.02, 0.31, 0.6 , 0.6 , 0.31, 0.31, 0.6, 0.5, 0.5, 0.5]
        self.B_vect_alt1 = (1 / np.asarray(K_vect_alt1)).tolist()
        self.d_vect_alt1 =  1.0
        self.n_vect_alt1 = 3.0
        self.cmax_alt1 = 6.0

        K_vect_alt2 = 0.5
        self.B_vect_alt1 = 1 / K_vect_alt2
        self.d_vect_alt2 = 1.0
        self.n_vect_alt2 = 3.0
        self.cmax_alt2 = 4.0

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
        # self.node_type_dict = None

        self.add_interactions = True

class QuadStateChain(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'QuadStateChain'
        self.short_name = 'QuadStateChain'

        self.N_nodes = 9
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ('G0', 'G1'), ('G1', 'G2'), ('G2', 'G0'),
                      ('G0', 'G0'), ('G2', 'G2'), ('G1', 'G1'),
                      ('S3', 'G0'), ('S4', 'G1'), ('S5', 'G2'),
                      ('H2', 'G0')
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.Is, EdgeType.Is, EdgeType.Is,
                           EdgeType.I, EdgeType.A, EdgeType.I,
                           EdgeType.A, EdgeType.A, EdgeType.A,
                           EdgeType.Is, EdgeType.Is, EdgeType.Is,
                           EdgeType.A
                      ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

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
        K_vect = [0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5,
                   0.25, 0.5, 1.0, 1.5, 0.5,
                   0.5, 0.5, 1.5,
                   0.5
                   ]

        self.B_vect = (1 / np.asarray(K_vect)).tolist()

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class FullQuadStateControl(LibNet):

    def __init__(self):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'FullQuadStateControl'
        self.short_name = 'FullQuadStateControl'

        # CASE TYPE QUADSTABLE with sensors and auxillary nodes in scale-free configuration:
        # Core is triangle loop with all auto-nodes edges:
        self.N_nodes = 20
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                 ('H2', 'G0'), ('H2', 'G1'), ('H2', 'G2'), ('H2', 'G3'), ('H2', 'G4'),
                 ('H0', 'G5'), ('H0', 'G6'), ('H0', 'G7'),
                 ('H1', 'G8'),
                 # ('F1', 'P1'), ('G4', 'P1'), ('P1', 'S2'),
                 ('F2', 'P2'), ('G1', 'P2'), ('P2', 'S0'),
                 # ('F3', 'P3'), ('G8', 'P3'), ('P3', 'S1'),
                 ]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.A, EdgeType.A,
                      EdgeType.I, EdgeType.I, EdgeType.I,
                      EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                      EdgeType.A, EdgeType.I, EdgeType.A,
                      EdgeType.A,
                      # EdgeType.A, EdgeType.I, EdgeType.A,
                      EdgeType.I, EdgeType.A, EdgeType.A,
                      # EdgeType.A, EdgeType.A, EdgeType.I,
                      ]

        self.node_type_dict = {'S': NodeType.sensor, 'P': NodeType.process, 'F': NodeType.factor}

        self.add_interactions = True