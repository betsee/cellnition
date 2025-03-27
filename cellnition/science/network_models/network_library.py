#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines multiple regulatory networks for general use.
'''
from abc import ABCMeta
import numpy as np
from cellnition.science.network_models.network_enums import EdgeType, NodeType
from cellnition._util.path.utilpathmake import FileRelative
from cellnition._util.path.utilpathself import get_data_csv_dir


class LibNet(object, metaclass=ABCMeta):
    '''

    '''
    def __init__(self):
        '''

        '''

        pass

    def count_nodes(self):

        # count the nodes based on the edges, just in case there's an error:
        nodes = []
        for ei, ej in self.edges:
            if ei not in nodes:
                nodes.append(ei)
            if ej not in nodes:
                nodes.append(ej)

        self.N_nodes = len(nodes)

class ActivatorExample(LibNet):
    '''
    Simple two-node example of an activating interaction.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'ActivatorExample'

        self.N_nodes = 2
        self.edges = [('S0', 'H0')]

        if activator_signals:
            self.edge_types = [EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class InhibitorExample(LibNet):
    '''
    Simple two-node example of an inhibiting interaction.
    '''

    def __init__(self, activator_signals: bool=False, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'InhibitorExample'

        self.N_nodes = 2
        self.edges = [('S0', 'H0')]

        if activator_signals:
            self.edge_types = [EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeChain(LibNet):
    '''
    Simple example of two-nodes in a chain with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChain'

        if one_input is False:
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
        else:
            self.N_nodes = 3
            self.edges = [('H0', 'H1'),
                     ('S0', 'H0')
                     ]

            if activator_signals:
                self.edge_types = [EdgeType.I,
                                   EdgeType.A,
                                   ]
            else:
                self.edge_types = [EdgeType.I,
                                   EdgeType.I,
                                   ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeDoubleChain(LibNet):
    '''
    Simple example of two-nodes in a double chain with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeDoubleChain'

        if one_input is False:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'),
                          ('H1', 'H0'),
                     ('S0', 'H0'), ('S1', 'H1')
                     ]

            if activator_signals:
                self.edge_types = [EdgeType.I, EdgeType.I,
                                   EdgeType.A, EdgeType.A
                                   ]
            else:
                self.edge_types = [EdgeType.I, EdgeType.I,
                                   EdgeType.I, EdgeType.I,
                                   ]
        else:
            self.N_nodes = 3
            self.edges = [('H0', 'H1'),
                          ('H1', 'H0'),
                     ('S0', 'H0')
                     ]

            if activator_signals:
                self.edge_types = [EdgeType.I, EdgeType.I,
                                   EdgeType.A,
                                   ]
            else:
                self.edge_types = [EdgeType.I, EdgeType.I,
                                   EdgeType.I,
                                   ]



        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeChainSelfLoop(LibNet):
    '''
    Simple example of two-nodes in a chain, with self loop on one node, and
     with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChainSelfLoop'

        if one_input is False:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'),
                     ('H0', 'H0'),
                     ('S0', 'H0'), ('S1', 'H1')
                     ]
        else:
            self.N_nodes = 3
            self.edges = [('H0', 'H1'),
                     ('H0', 'H0'),
                     ('S0', 'H0')
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
    '''
     Simple example of two-nodes in a chain, with self loops on both nodes, and
     with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeChainSelfLoops'

        if one_input is False:
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
                self.edge_types = [EdgeType.I,
                                   EdgeType.A, EdgeType.A,
                                   EdgeType.I, EdgeType.I
                                   ]
        else:
            self.N_nodes = 3
            self.edges = [('H0', 'H1'),
                     ('H0', 'H0'), ('H1', 'H1'),
                     ('S0', 'H0')
                     ]

            if activator_signals:
                self.edge_types = [EdgeType.I,
                                   EdgeType.A, EdgeType.A,
                                   EdgeType.A
                                   ]
            else:
                self.edge_types = [EdgeType.I,
                                   EdgeType.A, EdgeType.A,
                                   EdgeType.I
                                   ]



        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeCycle(LibNet):
    '''
     Simple example of two-nodes in a cycle,
     with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycle'

        if one_input is False:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('S0', 'H0'), ('S1', 'H1')
                     ]
        else:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('S0', 'H0')
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.A,
                          EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.A,
                          EdgeType.I, EdgeType.I
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class BinodeCycleSelfLoop(LibNet):
    '''
     Simple example of two-nodes in a cycle, with self loop on one node, and
     with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycleSelfLoop'

        if one_input is False:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('H0', 'H0'),
                     ('S0', 'H0'), ('S1', 'H1')
                     ]
        else:
            self.N_nodes = 3
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('H0', 'H0'),
                     ('S0', 'H0')
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
    '''
     Simple example of two-nodes in a cycle, with self loops on both nodes, and
     with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'BinodeCycleSelfLoops'

        if one_input is False:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('H0', 'H0'), ('H1', 'H1'),
                     ('S0', 'H0'), ('S1', 'H1')
                     ]
        else:
            self.N_nodes = 4
            self.edges = [('H0', 'H1'), ('H1', 'H0'),
                     ('H0', 'H0'), ('H1', 'H1'),
                     ('S0', 'H0')
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
     Simple example of three-nodes in a chain,
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChain'

        self.N_nodes = 6
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

class TrinodeDoubleChain(LibNet):
    '''
     Simple example of three-nodes in a double chain,
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeDoubleChain'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('H1', 'H0'), ('H2', 'H1'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChainSelfLoops(LibNet):
    '''
     Simple example of three-nodes in a chain, with self loops on each node, and
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChainSelfLoops'

        self.N_nodes = 6
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

class TrinodeForkSelfLoops(LibNet):
    '''
     Simple example of three-nodes in a forked configuration, with self loops on each node, and
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeForkSelfLoops'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H0', 'H2'),
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
class TrinodeFork(LibNet):
    '''
    Simple example of three-nodes in a forked configuration,
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeFork'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H0', 'H2'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeDoubleFork(LibNet):
    '''
    Simple example of three-nodes in a double forked configuration,
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = True):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeDoubleFork'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H0', 'H2'),
                      ('H1', 'H0'), ('H2', 'H0'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeChainFullyConnected(LibNet):
    '''
    Simple example of three-nodes in a double chain configuration, with
    self-loops on all internal nodes, and
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeChainFullyConnected'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('H1', 'H0'), ('H2', 'H1'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeForkFullyConnected(LibNet):
    '''
    Simple example of three-nodes in a double fork configuration, with
    self-loops on all internal nodes, and
     with input nodes acting on each internal node.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeForkFullyConnected'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H0', 'H2'),
                      ('H1', 'H0'), ('H2', 'H0'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCycle(LibNet):
    '''
    Simple example of three nodes in a cycle,
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycle'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.A,
                          EdgeType.I, EdgeType.I, EdgeType.I,
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class TrinodeDoubleCycle(LibNet):
    '''
    Simple example of three nodes in a double cycle,
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeDoubleCycle'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H1', 'H0'), ('H2', 'H1'), ('H0', 'H2'),
                 ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                 ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                               EdgeType.I, EdgeType.I, EdgeType.A,
                          EdgeType.I, EdgeType.I, EdgeType.I,
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class TrinodeCycleSelfLoops(LibNet):
    '''
    Simple example of three nodes in a cycle,
    with self-loops on all internal nodes, and
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycleSelfLoops'

        self.N_nodes = 6
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
                          EdgeType.I, EdgeType.I, EdgeType.I,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCycleFullyConnected(LibNet):
    '''
    Simple example of three nodes in a double cycle,
    with self-loops on all internal nodes, and
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=False):
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
            self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.I, EdgeType.A, EdgeType.A,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCycleFullyConnected2(LibNet):
    '''
    Simple example of three nodes in a double cycle,
    with self-loops on all internal nodes, and
    with input nodes acting on each internal node.
    (Not sure how this differs from TrinodeCycleFullyConnected,
    but keeping just in case we need it someday.)
    '''

    def __init__(self, activator_signals: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCycleFullyConnected2'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H1', 'H0'), ('H2', 'H1'), ('H0', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                     ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.I, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.I, EdgeType.A, EdgeType.A,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCyclesConnected(LibNet):
    '''
    Two three-node cycles connected together,
    with self-loops on three internal nodes, and
    with input nodes acting on three internal nodes.
    '''

    def __init__(self, activator_signals: bool=False):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'TrinodeCyclesConnected'

        self.N_nodes = 9
        self.edges = [('H0', 'H1'), ('H1', 'H2'), ('H2', 'H0'),
                      ('H1', 'H0'), ('H2', 'H1'), ('H0', 'H2'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H1', 'H1'),
                      ('H3', 'H4'), ('H4', 'H5'), ('H5', 'H3'),
                      ('H4', 'H3'), ('H5', 'H4'), ('H3', 'H5'),
                      ('H3', 'H3'), ('H5', 'H5'), ('H4', 'H4'),
                      ('H2', 'H3'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'),
                     ]

        if activator_signals:
            self.edge_types = [EdgeType.A for i in self.edges]
        else:
            self.edge_types = [EdgeType.A for i in self.edges]
            self.edge_types[-3:] = [EdgeType.I, EdgeType.I, EdgeType.I]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = Non
        self.add_interactions = True

class BasicQuadnodeNet(LibNet):
    '''
    Simple example of four nodes in a cycle,
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

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
    '''
    Simple example of four nodes in a cycle,
    with self-loops on each internal node, and
    with input nodes acting on each internal node.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

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
    '''
    Simple example of five nodes in a cycle,
    with input nodes acting on each internal node.
    '''

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
    '''
    Example of three nodes in a cycle, with many
    output nodes spanning from the core three nodes,
    and three input nodes acting on the three core nodes.
    '''

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
    '''
    Example of three nodes in a cycle, with many
    output nodes spanning from the core three nodes,
    and three input nodes acting on the three core nodes.
    This network has feedback between output and input nodes.
    '''

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
    '''
    Example of a simple network arrangement that can act as a
    homeostat by creating attractor limit cycles.
    '''

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
    '''
    Example of a simple network arrangement that can act as a
    homeostat by creating attractor limit cycles.
    '''

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
    '''
    Example of a simple network arrangement that can act as a
    homeostat by creating attractor limit cycles.
    '''

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
    '''
    Example of four nodes in a cycle, with many
    output nodes spanning from the core four nodes,
    and three input nodes acting on the four core nodes.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

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
    '''
    This biological network is the Oct4-Sox2-Nanog multistable core
    network of embryonic stem cells, with extrinsic signalling
    factors included.

    The network is sourced from the reference:
    Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
    cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

    '''

    def __init__(self, activator_signals: bool=True):
        '''


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
                      ('DAG', 'PKC')
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
                      ]

        self.node_type_dict = None

        self.add_interactions = True

class StemCellNet(LibNet):
    '''
    This biological network is the Oct4-Sox2-Nanog multistable core
    network of embryonic stem cells, with extrinsic signalling
    factors excluded.

    The network is sourced from the reference:
    Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
    cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

    '''

    def __init__(self, activator_signals: bool=True):
        '''


        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellNet'

        self.N_nodes = 6
        self.edges = [
                      ('NANOG', 'OCT4'),
                      ('OCT4', 'NANOG'),
                      ('OCT4', 'SOX2'),
                      ('SOX2', 'OCT4'),
                      ('SOX2', 'NANOG'),
                      ('NANOG', 'SOX2'),
                      ('SOX2', 'SOX2'),
                      ('OCT4', 'OCT4'),
                      ('S0', 'SOX2'),
                      ('S1', 'NANOG'),
                      ('S2', 'OCT4'),

                 ]

        if activator_signals:
            self.edge_types = [EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.A,
                               EdgeType.I,
                               EdgeType.I,
                               EdgeType.I,
                          ]
        else:
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
                               EdgeType.A,
                          ]

        self.node_type_dict = None

        self.add_interactions = True
        self.N_input_edges = 3

class StemCellTriad(LibNet):
    '''
    This biological network is based on the Oct4-Sox2-Nanog multistable core
    network of embryonic stem cells, with basic signalling factors included.
    Here we are using this network to explore different combinations and
    permutations of edge interaction types.
    '''

    def __init__(self, activator_signals: bool=True):
        '''


        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriad'

        self.N_nodes = 3
        self.edges = [
                      ('NANOG', 'OCT4'),
                      ('OCT4', 'NANOG'),
                      ('OCT4', 'SOX2'),
                      ('SOX2', 'OCT4'),
                      ('SOX2', 'NANOG'),
                      ('NANOG', 'SOX2'),
                      ('SOX2', 'SOX2'),
                      ('OCT4', 'OCT4'),
                 ]

        self.edge_types = [EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                      ]

        self.N_input_edges = 0

        self.node_type_dict = None

        self.add_interactions = True

class StemCellQuadrad(LibNet):
    '''
    This biological network is the Oct4-Sox2-Nanog + KLF4 multistable core
    network of embryonic stem cells, with some extrinsic signalling
    factors included.

    The network is sourced from the reference:
    Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
    cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

    '''

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellQuadrad'

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
                      ('KLF4', 'KLF4'),
                      ('KLF4', 'NANOG'),
                      ('KLF4', 'SOX2'),
                      ('KLF4', 'OCT4'),
                      ('S1', 'SOX2'),
                      ('S2', 'NANOG'),
                      ('S0', 'OCT4'),
                      # ('S3', 'KLF4'),

                 ]

        self.edge_types = [EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           # EdgeType.I,
                      ]

        self.N_input_edges = 3

        self.node_type_dict = None

        self.add_interactions = True

class StemCellTriadChain(LibNet):
    '''
    No matter which combination of activating or inhibiting edges were chosen,
    equilibrium states corresponding to the SOX2-OCT4-NANOG triad system could
    not be produced with a fully-connected cycle. We found that by breaking the
    cycle into a double chain, biorealistic equilibrium states could be produced.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriadChain'

        self.N_nodes = 6
        self.edges = [('OCT4', 'NANOG'), ('NANOG', 'SOX2'),
                      ('NANOG', 'OCT4'), ('SOX2', 'NANOG'),
                      ('OCT4', 'OCT4'), ('SOX2', 'SOX2'),
                      ('S0', 'OCT4'), ('S1', 'NANOG'), ('S2', 'SOX2')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class StemCellTriadChain0(LibNet):
    '''
    No matter which combination of activating or inhibiting edges were chosen,
    equilibrium states corresponding to the SOX2-OCT4-NANOG triad system could
    not be produced with a fully-connected cycle. We found that by breaking the
    cycle into a double chain, biorealistic equilibrium states could be produced.
    (Not sure how this differs from StemCellTriadChain; keeping just in case).
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriadChain0'

        self.N_nodes = 6
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('H1', 'H0'), ('H2', 'H1'),
                      ('H0', 'H0'), ('H2', 'H2'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class StemCellTriadChain2(LibNet):
    '''
    No matter which combination of activating or inhibiting edges were chosen,
    equilibrium states corresponding to the SOX2-OCT4-NANOG triad system could
    not be produced with a fully-connected cycle. We found that by breaking the
    cycle into a double chain, biorealistic equilibrium states could be produced.
    (Not sure how this differs from StemCellTriadChain; keeping just in case).
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriadChain2'

        self.N_nodes = 8
        self.edges = [('H0', 'H1'), ('H1', 'H2'),
                      ('H1', 'H0'), ('H2', 'H1'),
                      ('H0', 'H0'), ('H2', 'H2'), ('H3', 'H3'),
                      ('H3', 'H0'), ('H3', 'H1'), ('H3', 'H2'),
                      ('S0', 'H0'), ('S1', 'H1'), ('S2', 'H2'), ('S3', 'H3')
                      ]

        if activator_signals:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A, EdgeType.A,
                               ]
        else:
            self.edge_types = [EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I,
                               ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class AKTNet(LibNet):
    '''
    This biological network is the PI3K/AKT/mTOR (PAM) signaling transduction pathway.

    The network is sourced from the reference:

    Glaviano et al. PI3K/AKT/mTOR signaling transduction pathway and targeted therapies in cancer.
    Mol Cancer. 2023 Aug 18;22(1):138. doi: 10.1186/s12943-023-01827-6.

    The modelled network has been slightly simplified from the source reference by combining elements in direct
    signalling chains, and is the version that appeared in the first pre-print of our RNM paper.

    '''

    def __init__(self, activator_signals: bool=True):
        '''


        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'AKTNet'

        self.N_nodes = 24
        self.edges = [
                      # Input edges:
                      ('Growth_RTK', 'RAS'),
                      ('Growth_RTK', 'GAB12'),
                      ('Survival_RTK', 'PI3K'),
                      ('Wnt_Frizzled', 'Dsh'),

                      # RTK-Growth sub-pathway:
                      ('RAS', 'RAF'),
                      ('RAS', 'PI3K'),
                      ('GAB12', 'PI3K'),
                      ('RAF', 'ERK'),
                      ('ERK', 'TSCComplex'),
                      ('ERK', 'eIF4E'),

                      # RTK-survival sub-pathway:
                      ('PI3K', 'AKT'),
                      ('AKT', 'RAF'),
                      ('AKT', 'TSCComplex'),
                      ('AKT', 'FOXO'),
                      ('AKT', 'AxinComplex'),
                      ('AKT', 'bCAT'),
                      ('TSCComplex', 'mTORC1'),
                      ('mTORC1', 'EBP1'),
                      ('EBP1', 'eIF4E'),

                      # WNT-Frizzled sub-pathway
                      ('Dsh', 'AxinComplex'),
                      ('AxinComplex', 'bCAT'),

                      # Output edges
                      ('ERK', 'CellSurvival'),
                      ('eIF4E', 'Translation'),
                      ('mTORC1', 'CellCycle'),
                      ('mTORC1', 'CellSurvival'),
                      ('mTORC1', 'Metabolism'),
                      ('bCAT', 'Proliferation'),
                      ('bCAT', 'Proteosomes'),
                      ('FOXO', 'Apoptosis'),

                 ]

        self.edge_types = [
                          # Input edge types:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,

                           # RTK-Growth sub-pathway:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,

                          # RTK-survival sub-pathway:
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,

                         # WNT-Frizzled sub-pathway:
                           EdgeType.I,
                           EdgeType.A,

                         # Output edge types:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                      ]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3
class AKTNet1(LibNet):
    '''
    This biological network is the PI3K/AKT/mTOR (PAM) signaling transduction pathway.

    The network is sourced from the reference:

    Glaviano et al. PI3K/AKT/mTOR signaling transduction pathway and targeted therapies in cancer.
    Mol Cancer. 2023 Aug 18;22(1):138. doi: 10.1186/s12943-023-01827-6.

    The modelled network has been slightly simplified from the source reference by combining elements in direct
    signalling chains and by removing some cross-talk connections between pathways in order to
    prevent nonsense output.

    '''

    def __init__(self, activator_signals: bool=True):
        '''


        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'AKTNet1'

        self.N_nodes = 26
        self.edges = [
                      # Input edges:
                      ('Growth_RTK', 'RAS'),
                      ('Survival_RTK', 'PI3K'),
                      ('Wnt_Frizzled', 'Dsh'),

                      # RTK-Growth sub-pathway:
                      ('RAS', 'RAF'),
                      ('RAF', 'MEK'),
                      ('MEK', 'ERK'),
                      ('ERK', 'MNK1'),
                      ('ERK', 'RSK'),
                      ('ERK', 'mTORC1'),
                      ('ERK', 'TSCComplex'),
                      ('RSK', 'mTORC1'),
                      ('MNK1', 'eIF4E'),

                      # RTK-survival sub-pathway:
                      ('PI3K', 'AKT'),
                      ('AKT', 'RAF'),
                      ('AKT', 'TSCComplex'),
                      ('AKT', 'FOXO'),
                      ('AKT', 'AxinComplex'),
                      ('AKT', 'bCAT'),
                      ('TSCComplex', 'mTORC1'),
                      ('mTORC1', 'EBP1'),
                      ('EBP1', 'eIF4E'),

                      # WNT-Frizzled sub-pathway
                      ('Dsh', 'AxinComplex'),
                      ('AxinComplex', 'bCAT'),

                      # Output edges
                      ('ERK', 'CellSurvival'),
                      ('eIF4E', 'Translation'),
                      ('mTORC1', 'CellCycle'),
                      ('mTORC1', 'Metabolism'),
                      ('bCAT', 'Proliferation'),
                      ('bCAT', 'Proteosomes'),
                      ('FOXO', 'Apoptosis'),

                 ]

        self.edge_types = [
                          # Input edge types:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,

                           # RTK-Growth sub-pathway:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,

                          # RTK-survival sub-pathway:
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,

                         # WNT-Frizzled sub-pathway:
                           EdgeType.I,
                           EdgeType.A,

                         # Output edge types:
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                      ]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3
class hESC_9a(LibNet):
    '''
    Uses the DNase Footprint derived TF networks at regulatory_networks.org to generate
    more structure around the classic SOX2-OCT4-NANOG triad.
    '''

    def __init__(self, activator_signals: bool=True, use_special_edges: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'hESC_9a'

        self.N_nodes = 12
        SOX2 = 'SOX2'
        POU5F1 = 'OCT4'
        NANOG = 'NANOG'
        KLF4 = 'KLF4'
        KLF15 = 'KLF15'
        SP1 = 'SP1'
        SP2 = 'SP2'
        SP3 = 'SP3'
        EGR2 = 'EGR2'


        self.edges = [
            (EGR2, KLF15),
            (EGR2, KLF4),
            (EGR2, POU5F1),
            (EGR2, SOX2),
            (EGR2, SP1),
            (EGR2, SP2),
            (EGR2, SP3),
            (KLF15, KLF4),
            (KLF15, POU5F1),
            (KLF15, SOX2),
            (KLF15, SP1),
            (KLF15, SP2),
            (KLF15, SP3),
            (KLF4, KLF15),
            (KLF4, KLF4),
            (KLF4, NANOG),
            (KLF4, POU5F1),
            (KLF4, SOX2),
            (KLF4, SP1),
            (NANOG, POU5F1),
            (NANOG, SOX2),
            (POU5F1, EGR2),
            (POU5F1, NANOG),
            (POU5F1, POU5F1),
            (POU5F1, SOX2),
            (SOX2, NANOG),
            (SOX2, POU5F1),
            (SP1, EGR2),
            (SP1, KLF15),
            (SP1, KLF4),
            (SP1, POU5F1),
            (SP1, SOX2),
            (SP1, SP1),
            (SP1, SP2),
            (SP1, SP3),
            (SP2, EGR2),
            (SP2, KLF15),
            (SP2, KLF4),
            (SP2, POU5F1),
            (SP2, SOX2),
            (SP2, SP1),
            (SP2, SP3),
            (SP3, EGR2),
            (SP3, KLF15),
            (SP3, KLF4),
            (SP3, POU5F1),
            (SP3, SOX2),
            (SP3, SP1),
            (SP3, SP2),
            (SOX2, SOX2),
            (SP3, SP3),
            (SP2, SP2),
            (KLF15, KLF15),
            ("S0", NANOG),
            ("S1", SOX2),
            ("S2", KLF4)
                 ]


        if use_special_edges is False:

            if activator_signals:
                self.edge_types = [EdgeType.I for i in self.edges]
                self.edge_types[-7:] = [EdgeType.A, EdgeType.A, EdgeType.A]

            else:
                self.edge_types = [EdgeType.A for i in self.edges]
                self.edge_types[-3:] = [EdgeType.I, EdgeType.I, EdgeType.I]

        else:

            self.edge_types = [EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.A,
                                EdgeType.I,
                                EdgeType.I,
                                EdgeType.I]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3

class hESC_9b(LibNet):
    '''
    Uses the DNase Footprint derived TF networks at regulatory_networks.org to generate
    more structure around the classic SOX2-OCT4-NANOG triad.

    '''

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'hESC_9b'

        self.N_nodes = 12
        SOX2 = 'SOX2'
        POU5F1 = 'OCT4'
        NANOG = 'NANOG'
        KLF4 = 'KLF4'
        KLF15 = 'KLF15'
        SP1 = 'SP1'
        SP2 = 'SP2'
        SP3 = 'SP3'
        EGR2 = 'EGR2'


        self.edges = [
            (EGR2, KLF15),
            (EGR2, KLF4),
            (EGR2, POU5F1),
            (EGR2, SOX2),
            (EGR2, SP1),
            (EGR2, SP2),
            (EGR2, SP3),
            (KLF15, KLF15),
            (KLF15, KLF4),
            (KLF15, POU5F1),
            (KLF15, SOX2),
            (KLF15, SP1),
            (KLF15, SP2),
            (KLF15, SP3),
            (KLF4, KLF15),
            (KLF4, KLF4),
            (KLF4, NANOG),
            (KLF4, POU5F1),
            (KLF4, SOX2),
            (KLF4, SP1),
            (NANOG, POU5F1),
            (NANOG, SOX2),
            (POU5F1, EGR2),
            (POU5F1, NANOG),
            (POU5F1, POU5F1),
            (POU5F1, SOX2),
            (SOX2, NANOG),
            (SOX2, POU5F1),
            (SOX2, SOX2),
            (SP1, EGR2),
            (SP1, KLF15),
            (SP1, KLF4),
            (SP1, POU5F1),
            (SP1, SOX2),
            (SP1, SP1),
            (SP1, SP2),
            (SP1, SP3),
            (SP2, EGR2),
            (SP2, KLF15),
            (SP2, KLF4),
            (SP2, POU5F1),
            (SP2, SOX2),
            (SP2, SP1),
            (SP2, SP2),
            (SP2, SP3),
            (SP3, EGR2),
            (SP3, KLF15),
            (SP3, KLF4),
            (SP3, POU5F1),
            (SP3, SOX2),
            (SP3, SP1),
            (SP3, SP2),
            (SP3, SP3),
            ("S0", POU5F1),
            ("S1", SOX2),
            ("S2", NANOG)
                 ]

        self.edge_types = [EdgeType.A for i in self.edges]
        self.edge_types[-3:] = [EdgeType.I, EdgeType.I, EdgeType.I]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3

class hESC_9(LibNet):
    '''
    Uses the DNase Footprint derived TF networks at regulatory_networks.org to generate
    more structure around the classic SOX2-OCT4-NANOG triad.
    '''

    def __init__(self, activator_signals: bool=True):
        '''

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'hESC_9'

        self.N_nodes = 12
        SOX2 = 'SOX2'
        POU5F1 = 'OCT4'
        NANOG = 'NANOG'
        KLF4 = 'KLF4'
        KLF15 = 'KLF15'
        SP1 = 'SP1'
        SP2 = 'SP2'
        SP3 = 'SP3'
        EGR2 = 'EGR2'

        self.edges = [
            (EGR2, KLF15),
            (EGR2, KLF4),
            (EGR2, POU5F1),
            (EGR2, SOX2),
            (EGR2, SP1),
            (EGR2, SP2),
            (EGR2, SP3),
            (KLF15, KLF15),
            (KLF15, KLF4),
            (KLF15, POU5F1),
            (KLF15, SOX2),
            (KLF15, SP1),
            (KLF15, SP2),
            (KLF15, SP3),
            (KLF4, KLF15),
            (KLF4, KLF4),
            (KLF4, NANOG),
            (KLF4, POU5F1),
            (KLF4, SOX2),
            (KLF4, SP1),
            (NANOG, POU5F1),
            (NANOG, SOX2),
            (POU5F1, EGR2),
            (POU5F1, NANOG),
            (POU5F1, POU5F1),
            (POU5F1, SOX2),
            (SOX2, NANOG),
            (SOX2, POU5F1),
            (SOX2, SOX2),
            (SP1, EGR2),
            (SP1, KLF15),
            (SP1, KLF4),
            (SP1, POU5F1),
            (SP1, SOX2),
            (SP1, SP1),
            (SP1, SP2),
            (SP1, SP3),
            (SP2, EGR2),
            (SP2, KLF15),
            (SP2, KLF4),
            (SP2, POU5F1),
            (SP2, SOX2),
            (SP2, SP1),
            (SP2, SP2),
            (SP2, SP3),
            (SP3, EGR2),
            (SP3, KLF15),
            (SP3, KLF4),
            (SP3, POU5F1),
            (SP3, SOX2),
            (SP3, SP1),
            (SP3, SP2),
            (SP3, SP3),
            ("S2", NANOG),
            ("S1", SOX2),
            ("S3", KLF4),
            ("S0", POU5F1)
                 ]

        self.edge_types = [EdgeType.A for i in self.edges]
        self.edge_types[-4:] = [EdgeType.I, EdgeType.I, EdgeType.I, EdgeType.I]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 4

class MAPK_net(LibNet):
    '''
    This method imports the MAPK cancer cell fate network developed by
    Grieco et al. "Integrative Modelling of the Influence of MAPK Network on
    Cancer Cell Fate Decision." PLoS Comp. Bio. 9(10): e1003286. 2013
    https://doi.org/10.1371/journal.pcbi.1003286
    '''

    def __init__(self):
        '''

        '''

        super().__init__()

        self.name = 'MAPK_Cancer'

        # nodes from the reference:
        self.nodes_list = ['AKT',
                            'AP1',
                            'Apoptosis',
                            'ATF2',
                            'ATM',
                            'BCL2',
                            'CREB',
                            'DNA_damage',
                            'DUSP1',
                            'EGFR',
                            'EGFR_stimulus',
                            'ELK1',
                            'ERK',
                            'FGFR3',
                            'FGFR3_stimulus',
                            'FOS',
                            'FOXO3',
                            'FRS2',
                            'GAB1',
                            'GADD45',
                            'GRB2',
                            'Growth_Arrest',
                            'JNK',
                            'JUN',
                            'MAP3K1_3',
                            'MAX',
                            'MDM2',
                            'MEK1_2',
                            'MSK',
                            'MTK1',
                            'MYC',
                            'p14',
                            'p21',
                            'p38',
                            'p53',
                            'p70',
                            'PDK1',
                            'PI3K',
                            'PKC',
                            'PLCG',
                            'PPP2CA',
                            'Proliferation',
                            'PTEN',
                            'RAF',
                            'RAS',
                            'RSK',
                            'SMAD',
                            'SOS',
                            'SPRY',
                            'TAK1',
                            'TAOK',
                            'TGFBR',
                            'TGFBR_stimulus']

        self.N_nodes = len(self.nodes_list)
        self.node_type_dict = None

        # As this network is very large, here we create the edges and edge types using a
        # signed adjacency matrix acquired from Cell Collective.

        # Path to load the signed adjacency matrix:
        CSV_DIR = get_data_csv_dir()
        csv_fname = FileRelative(CSV_DIR, 'MAPK_Cancer_Mo.csv')

        # Load the csv file as a numpy matrix:
        self._A_o = np.genfromtxt(csv_fname, delimiter=',', dtype=np.int32)

        # Generate edges and edge types using a signed adjacency matrix:
        self.edges = []
        self.edge_types = []
        for nde_i, nme_i in enumerate(self.nodes_list):
            for nde_j, nme_j in enumerate(self.nodes_list):
                a_ij = self._A_o[nde_i, nde_j]
                if a_ij != 0:
                    self.edges.append((nme_j, nme_i))
                    if a_ij == 1:
                        self.edge_types.append(EdgeType.A)
                    elif a_ij == -1:
                        self.edge_types.append(EdgeType.I)
