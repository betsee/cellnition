#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines several types of study networks.
'''
from abc import ABCMeta
import numpy as np
from cellnition.science.network_models.network_enums import EdgeType, NodeType


class LibNet(object, metaclass=ABCMeta):
    '''

    '''
    def __init__(self):
        '''

        '''

        pass

class BinodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeNet'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('H0', 'H0'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BasicBinodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BasicBinodeNet'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.Is, EdgeType.Is
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeNet'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BasicTrinodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BasicTrinodeNet'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class BasicQuadnodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BasicQuadnodeNet'

        self.N_nodes = 8
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is, EdgeType.Is
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class QuadnodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'QuadnodeNet'

        self.N_nodes = 8
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'H0'),
                      ('H0', 'H0'), ('H1', 'H1'), ('H2', 'H2'), ('H3', 'H3'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3')
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                              ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.Is, EdgeType.Is, EdgeType.Is, EdgeType.Is
                              ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class PentanodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'PentanodeNet'

        self.N_nodes = 10
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'H4'), ('H4', 'H0'),
                 ('H0', 'H0'), ('H1', 'H1'), ('H2', 'H2'), ('H3', 'H3'), ('H4', 'H4'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3'), ('S4', 'H4')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,EdgeType.Is, EdgeType.Is
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeNetLoaded(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeNet_Loaded'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H1', 'H0'), ('H2', 'H1'), ('H0', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                     ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChain(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'QuadStateChain'

        self.N_nodes = 12
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ('G0', 'G1'), ('G1', 'G2'), ('G2', 'G0'),
                      ('G0', 'G0'), ('G2', 'G2'), ('G1', 'G1'),
                      ('S3', 'G0'), ('S4', 'G1'), ('S5', 'G2'),
                      ('H2', 'G0')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A
                          ]

        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.Is, EdgeType.Is, EdgeType.Is,
                               EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class FullTrinodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'FullTrinodeNet'

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

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                          EdgeType.A, EdgeType.I, EdgeType.A,
                          EdgeType.A
                          ]

        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                          EdgeType.A, EdgeType.I, EdgeType.A,
                          EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'G': NodeType.effector,
                               'H': NodeType.core}

        self.add_interactions = True

class BiLoopControlNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BiLoopControlNet'

        self.N_nodes = 7
        self.edges = [('S0', 'H0'), ('H0', 'E0'), ('E0', 'P0'), ('P0', 'S0'),
                      ('S0', 'H1'), ('H1', 'E1'), ('E1', 'P0'),
                      ('F0', 'P0')]

        self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.I, EdgeType.A,
                           EdgeType.I, EdgeType.A, EdgeType.A,
                           EdgeType.A
                           ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'E': NodeType.effector,
                               'P': NodeType.process,
                               'F': NodeType.factor
                               }

class BiLoopControlNet2(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BiLoopControlNet2'

        self.N_nodes = 7
        self.edges = [('S0', 'H0'), ('H0', 'E0'), ('E0', 'P0'), ('P0', 'S0'),
                      ('S0', 'H1'), ('H1', 'E1'), ('E1', 'P0'),
                      ('H0', 'H1'), ('H1', 'H0'),
                      ('F0', 'P0')]

        self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.I, EdgeType.A,
                           EdgeType.I, EdgeType.A, EdgeType.A,
                           EdgeType.I, EdgeType.I,
                           EdgeType.A
                           ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'E': NodeType.effector,
                               'P': NodeType.process,
                               'F': NodeType.factor
                               }

class BiLoopControlNet3(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BiLoopControlNet3'

        self.N_nodes = 7
        self.edges = [('S0', 'H0'), ('H0', 'E0'), ('E0', 'P0'), ('P0', 'S0'),
                      ('S0', 'H1'), ('H1', 'E1'), ('E1', 'P0'),
                      # ('H0', 'H1'), ('H1', 'H0'),
                      ('F0', 'P0')]

        self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I, EdgeType.I,
                           EdgeType.I, EdgeType.I, EdgeType.A,
                           # EdgeType.I, EdgeType.I,
                           EdgeType.A
                           ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'E': NodeType.effector,
                               'P': NodeType.process,
                               'F': NodeType.factor
                               }

class FullQuadnodeNet(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'FullQuadnodeNet'

        self.N_nodes = 16
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H3'), ('H3', 'H0'),
                      ('H0', 'H0'), ('H1', 'H1'), ('H2', 'H2'), ('H3', 'H3'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3'),
                      ('H0', 'G0'), ('H0', 'G1'), ('H0', 'G2'), ('H1', 'G3'),
                      ('H1', 'G4'), ('H2', 'G5'), ('H2', 'G6'), ('H3', 'G7'),
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.I, EdgeType.A,
                               EdgeType.I, EdgeType.A, EdgeType.I, EdgeType.A,
                              ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.Is, EdgeType.Is, EdgeType.Is, EdgeType.Is,
                               EdgeType.A, EdgeType.A, EdgeType.I, EdgeType.A,
                               EdgeType.I, EdgeType.A, EdgeType.I, EdgeType.A,
                              ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'G': NodeType.effector,
                               'H': NodeType.core}

        self.add_interactions = True