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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters. This is the monostable example in our first paper.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters. This is the monostable example in our first paper.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

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

    def __init__(self, activator_signals: bool=True, one_input: bool=False):
        '''
        This is indicated to be a tri-stable network. It is used for
        parameter space searches to look for further states with changes
        to parameters.

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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
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
            self.edge_types = [EdgeType.I, EdgeType.A, EdgeType.I,
                          EdgeType.A, EdgeType.A, EdgeType.A,
                          ]
        else:
            self.edge_types = [EdgeType.Is, EdgeType.A, EdgeType.Is,
                          EdgeType.Is, EdgeType.Is, EdgeType.Is,
                          ]

        self.node_type_dict = {'S': NodeType.signal}

        self.add_interactions = True

class TrinodeDoubleCycle(LibNet):

    def __init__(self, activator_signals: bool=True):
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
            self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
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
            self.edge_types = [EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                               EdgeType.A, EdgeType.A, EdgeType.A,
                          EdgeType.I, EdgeType.I, EdgeType.I,
                          ]

        self.node_type_dict = {'S': NodeType.signal}
        # self.node_type_dict = None

        self.add_interactions = True

class TrinodeCyclesConnected(LibNet):

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
                      ('S0', 'SOX2'),
                      ('S1', 'NANOG'),
                      ('S2', 'NANOG'),

                 ]

        self.edge_types = [EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                      ]

        self.node_type_dict = None

        self.add_interactions = True
        self.N_input_edges = 3

class StemCellTriad(LibNet):
    '''

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog multistable core
        network of embryonic stem cells, with basic signalling factors included.
        Here we are using this network to explore different combinations and
        permutations of edge interaction types.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriad'

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
                      ('S1', 'SOX2'),
                      ('S2', 'NANOG'),
                      ('S0', 'OCT4')
                 ]

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
                           EdgeType.I
                      ]

        self.N_input_edges = 3

        self.node_type_dict = None

        self.add_interactions = True

class StemCellQuadrad(LibNet):
    '''

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the Oct4-Sox2-Nanog + KLF4 multistable core
        network of embryonic stem cells, with extrinsic signalling
        factors included.

        The network is sourced from the reference:
        Mossahbi-Mohammadi, M. et al. FGF signalling pathway: A key regulator of stem
        cell pluripotency. Frontiers in Cell and Developmental Biology. 8:79. 2020.

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
    Example of the lowest hierarchically incoherent network
    for a 3-node network.
    '''
    # Initialize the superclass:

    def __init__(self, activator_signals: bool = False):

        # Initialize the superclass:
        super().__init__()

        self.name = 'StemCellTriadChain'

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
    Example of the lowest hierarchically incoherent network
    for a 3-node network. With KLF4
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

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        This biological network is the PI3K/AKT/mTOR signaling transduction pathway.

        The network is sourced from the reference:

        Glaviano et al. PI3K/AKT/mTOR signaling transduction pathway and targeted therapies in cancer.
        Mol Cancer. 2023 Aug 18;22(1):138. doi: 10.1186/s12943-023-01827-6.

        The modelled network has been simplified from the source reference by combining elements in direct
        signalling chains.

        '''
        # Initialize the superclass:
        super().__init__()

        self.name = 'AKTNet'

        self.N_nodes = 25
        self.edges = [
                      ('Growth-RTK', 'RAS'),
                      ('Growth-RTK', 'GAB1/2'),
                      ('Survival-RTK', 'PI3K'),
                      ('Wnt-Frizzled', 'Dsh'),
                      ('RAS', 'RAF'),
                      ('RAS', 'PI3K'),
                      ('GAB1/2', 'PI3K'),
                      ('PI3K', 'AKT'),
                      ('AKT', 'RAF'),
                      ('AKT', 'TSCComplex'),
                      ('AKT', 'FOXO'),
                      ('AKT', 'AxinComplex'),
                      ('AKT', 'bCAT'),
                      ('Dsh', 'AxinComplex'),
                      ('AxinComplex', 'bCAT'),
                      ('RAF', 'ERK'),
                      ('ERK', 'TSCComplex'),
                      ('ERK', 'mTORC1'),
                      ('ERK', 'eIF4E'),
                      ('TSCComplex', 'mTORC1'),
                      ('mTORC1', '4EBP1'),
                      ('4EBP1', 'eIF4E'),
                      ('ERK', 'Cell Survival'),
                      ('eIF4E', 'Translation'),
                      ('mTORC1', 'Cell Survival'),
                      ('mTORC1', 'Cell Cycle'),
                      ('mTORC1', 'Metabolism'),
                      ('bCAT', 'Proliferation'),
                      ('bCAT', 'Proteosome'),
                      ('FOXO', 'Apoptosis'),

                 ]

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
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.I,
                           EdgeType.I,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A,
                           EdgeType.A
                      ]

        self.effector_edges = [('ERK', 'Cell Survival'),
                               ('mTORC1', 'Cell Survival'),
                               ('mTORC1', 'Cell Cycle'),
                               ('mTORC1', 'Metabolism'),
                               ('FOXO', 'Apoptosis'),
                               ('bCAT', 'Proliferation'),
                               ('bCAT', 'Proteasome'),
                               ('eIF4E', 'Translation'),]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3

class hESC_9a(LibNet):
    '''
    Works from the DNase Footprint derived TF networks to generate
    more structure around the classic SOX2-OCT4-NANOG triad.

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        Works from the DNase Footprint derived TF networks to generate
        more structure around the classic SOX2-OCT4-NANOG triad.

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
            ("S0", NANOG),
            ("S1", SOX2),
            ("S2", KLF4)
                 ]

        self.edge_types = [EdgeType.A for i in self.edges]
        self.edge_types[-3:] = [EdgeType.I, EdgeType.I, EdgeType.I]

        self.node_type_dict = None

        self.add_interactions = True

        self.N_input_edges = 3

class hESC_9b(LibNet):
    '''
    Works from the DNase Footprint derived TF networks to generate
    more structure around the classic SOX2-OCT4-NANOG triad.

    '''

    def __init__(self, activator_signals: bool=True):
        '''
        Works from the DNase Footprint derived TF networks to generate
        more structure around the classic SOX2-OCT4-NANOG triad.

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
    Works from the DNase Footprint derived TF networks to generate
    more structure around the classic SOX2-OCT4-NANOG triad.
    This network allows inputs on all Yamanka factors (SOX2, OCT4, NANOG and KLF4)
    '''

    def __init__(self, activator_signals: bool=True):
        '''
        Works from the DNase Footprint derived TF networks to generate
        more structure around the classic SOX2-OCT4-NANOG triad.

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
