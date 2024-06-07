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

class BinodeChain(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChain'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I,
                          EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeChainSelfLoop(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChainSelfLoop'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'),
                 ('H0', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I,
                          EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.Is,
                          EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeChainSelfLoops(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChainSelfLoops'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'),
                 ('H0', 'H0'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.Is,
                          EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeCycle(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycle'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                          EdgeType.A, EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeCycleSelfLoop(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycleSelfLoop'

        self.N_nodes = 4
        self.edges = [('H0', 'H1'), ('H1', 'H0'),
                 ('H0', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                          EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                          EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeCycleSelfLoops(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycleSelfLoops'

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
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                          EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChain(LibNet):
    '''
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChain'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChainSelfLoops(LibNet):
    '''
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChainSelfLoops'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChainFullyConnected(LibNet):
    '''
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChainFullyConnected'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('H1', 'H0'), ('H2', 'H1'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                               EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.Is,
                               EdgeType.Is, EdgeType.Is,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCycle(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycle'

        self.N_nodes = 3
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.A, EdgeType.Is,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class TrinodeCycleSelfLoops(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycleSelfLoops'

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
            self.edge_types = [EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCycleFullyConnected(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycleFullyConnected'

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
            self.edge_types = [EdgeType.Is, EdgeType.Is, EdgeType.Is,
                               EdgeType.Is, EdgeType.Is, EdgeType.Is,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

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

class FullTrinodeNetFeedback(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'FullTrinodeNetFeedback'

        # CASE TYPE QUADSTABLE with sensors and auxillary nodes in scale-free configuration:
        # Core is triangle loop with all auto-nodes edges:
        self.N_nodes = 9
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                 ('F0', 'H0'), ('F1', 'H1'), ('F2', 'H2'),
                 # ('H2', 'G0'), ('H2', 'G1'), ('H2', 'G2'), ('H2', 'G3'), ('H2', 'G4'),
                 # ('H0', 'G5'), ('H0', 'G6'), ('H0', 'G7'),
                 # ('H1', 'G8')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          # EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                          # EdgeType.A, EdgeType.I, EdgeType.A,
                          # EdgeType.A
                          ]

        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          # EdgeType.A, EdgeType.I, EdgeType.A, EdgeType.A, EdgeType.I,
                          # EdgeType.A, EdgeType.I, EdgeType.A,
                          # EdgeType.A
                          ]

        self.node_type_dict = {'S': NodeType.sensor,
                               'F': NodeType.factor,
                               'G': NodeType.effector,
                               'H': NodeType.core}

        self.add_interactions = True

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
                               'F': NodeType.factor,
                               'H': NodeType.core
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

class StemCellNetFull(LibNet):

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog multistable core
        network of embryonic stem cells, with extrinsic signalling
        factors included.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellNetFull'

        self.N_nodes = 27
        self.edges = [('FGF2', 'RAS'),
                      ('FGF2', 'PLCg'),
                      ('FGF2', 'PI3k'),
                      ('RAS', 'RAF'),
                      ('RAF', 'MEK1/2'),
                      ('MEK1/2', 'ERK1/2'),
                      ('ERK1/2', 'TBX3'),
                      ('TBX3', 'NANOG'),
                      ('NANOG', 'OCT4'),
                      ('OCT4', 'SOX2'),
                      ('SOX2', 'NANOG'),
                      ('OCT4', 'NANOG'),
                      ('NANOG', 'SOX2'),
                      ('SOX2', 'OCT4'),
                      ('SOX2', 'SOX2'),
                      ('OCT4', 'OCT4'),
                      ('PLCg', 'DAG'),
                      ('PKC', 'GSK3b'),
                      ('GSK3b', 'cMYC'),
                      ('cMYC', 'SOX2'),
                      ('IGF2', 'PIP3'),
                      ('PIP3', 'PKD1'),
                      ('PKD1', 'AKT'),
                      ('AKT', 'GSK3b'),
                      ('BMP4', 'SMAD1584'),
                      ('SMAD1584', 'NANOG'),
                      ('TGF', 'SMAD234'),
                      ('SMAD234', 'NANOG'),
                      ('WNT', 'DVL'),
                      ('DVL', 'bCAT'),
                      ('bCAT', 'TCF3'),
                      ('TCF3', 'NANOG'),
                      ('PI3k', 'PIP3'),
                      ('DAG', 'PKC'),
                      ('NANOG', 'NANOG')
                 ]

        self.edge_types = [EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A
                      ]

        self.node_type_dict = None

        self.add_interactions = True

class StemCellNet(LibNet):
    '''

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog multistable core
        network of embryonic stem cells, with extrinsic signalling
        factors included.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellNet'

        self.N_nodes = 8
        self.edges = [
                      ('NANOG', 'OCT4'),
                      ('OCT4', 'NANOG'),
                      ('OCT4', 'SOX2'),
                      ('SOX2', 'OCT4'),
                      ('SOX2', 'NANOG'),
                      ('NANOG', 'SOX2'),
                      ('SOX2', 'SOX2'),
                      ('OCT4', 'OCT4'),
                      ('NANOG', 'NANOG'),
                      ('cMYC', 'SOX2'),
                      ('TBX3', 'NANOG'),
                      ('BMP4', 'NANOG'),
                      ('TFG', 'NANOG'),
                      ('TCF3', 'NANOG')

                 ]

        self.edge_types = [EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                      ]

        self.node_type_dict = None

        self.add_interactions = True

class StemCellNet2(LibNet):
    '''

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog multistable core
        network of embryonic stem cells, with extrinsic signalling
        factors included.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellNet2'

        self.N_nodes = 15
        self.edges = [
                      ('NANOG', 'OCT4'),
                      ('OCT4', 'NANOG'),
                      ('OCT4', 'SOX2'),
                      ('SOX2', 'OCT4'),
                      ('SOX2', 'NANOG'),
                      ('NANOG', 'SOX2'),
                      ('SOX2', 'SOX2'),
                      ('OCT4', 'OCT4'),
                      ('NANOG', 'NANOG'),
                      ('cMYC', 'SOX2'),
                      ('TBX3', 'NANOG'),
                      ('BMP4', 'NANOG'),
                      ('TFG', 'NANOG'),
                      ('TCF3', 'NANOG'),
                      ('FGF2', 'TBX3'),
                      ('FGF2', 'GSK3b'),
                      ('GSK3b', 'cMYC'),
                      ('FGF2', 'PI3K'),
                      ('PI3K', 'PIP3'),
                      ('PIP3', 'AKT'),
                      ('AKT', 'GSK3b'),
                      ('IGF2', 'PIP3')

                 ]

        self.edge_types = [EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                      ]

        self.node_type_dict = None

        self.add_interactions = True

class AKTNet(LibNet):
    '''

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog multistable core
        network of embryonic stem cells, with extrinsic signalling
        factors included.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'AKTNet'

        self.N_nodes = 18
        self.edges = [
                      ('GrowthFactors', 'RAS'),
                      ('SurvivalFactors', 'PI3K'),
                      ('WNT', 'Dsh'),
                      ('RAS', 'RAF'),
                      ('RAF', 'MEK'),
                      ('MEK', 'ERK'),
                      ('Dsh', 'AxinComplex'),
                      ('ERK', 'eIF4E'),
                      ('ERK', 'mTORC1'),
                      ('mTORC1', '4EBP1'),
                      ('4EBP1', 'eIF4E'),
                      ('ERK', 'TSCComplex'),
                      ('AKT', 'TSCComplex'),
                      ('AKT', 'RAF'),
                      ('RAS', 'PI3K'),
                      ('PI3K', 'mTORC2'),
                      ('mTORC2', 'AKT'),
                      ('AKT', 'FOXO'),
                      ('AKT', 'AxinComplex'),
                      ('AxinComplex', 'bCAT'),
                      ('AKT', 'bCAT')
                 ]

        self.edge_types = [EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                      ]

        # self.effector_edges = [('ERK', 'Cell Survival'),
        #                        ('mTORC1', 'Cell Survival'),
        #                        ('mTORC1', 'Cell Cycle'),
        #                        ('mTORC1', 'Metabolism'),
        #                        ('FOXO', 'Apoptosis'),
        #                        ('bCAT', 'Proliferation'),
        #                        ('bCAT', 'Proteasome')]

        self.node_type_dict = None

        self.add_interactions = True

