#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module is an abstract base class for network models.
'''
from abc import ABCMeta, abstractmethod
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.network_models.network_enums import EdgeType, GraphType, NodeType, InterFuncType
from cellnition.science.network_models.interaction_functions import (f_neut_s)
import pygraphviz as pgv

# TODO: Add in stochasticity
# TODO: Time signals should be able to do flipped case (from 1 to zero)

class NetworkABC(object, metaclass=ABCMeta):
    '''
    This class allows one to generate a network using a random construction
    algorithm, or from user-input edges. It then performs analysis on the
    resulting graph to determine cycles, input and output degree distributions,
    and hierarchical attributes. The class then enables the user to build an
    analytical (i.e. symbolic math) module using the network, and has various routines
    to determine equilibrium points and stable states of the network model. The
    class then allows for time simulation of the network model as a dynamic system.

    Public Attributes
    -----------------
    GG = nx.DiGraph(self.edges_list)
    N_nodes = N_nodes # number of nodes in the network (as defined by user initially)
    nodes_index = self.nodes_list
    nodes_list = sorted(self.GG.nodes())
    N_edges = len(self.edges_list)
    edges_index = self.edges_list
    edges_list
    '''

    def __init__(self, N_Nodes: int
                 ):
        '''
        Initialize the class and build and characterize a network.

        Parameters
        -----------
        N_nodes: int
            The number of nodes to build the network (only used in randomly built networks, otherwise the number of
            nodes is calculated from the number of unique nodes supplied in the edges list).

        edges: list|ndarray|None = None
            A list of tuples that defines edges of a network, where each directed edge is a pair of nodes. The
            nodes may be integers or strings, but cannot be mixed type. If edges is left as None, then a
            graph will be randomly constructed.

        graph_type: GraphType = GraphType.scale_free
            The type of graph to generate in randomly-constructed networks.

        b_param: float = 0.20
            For scale-free randomly-constructed networks, this determines the amount of interconnectivity between
            the in and out degree distributions, and in practical terms, increases the number of cycles in the graph.
            Note that 1 - beta - gamma must be greater than 0.0.

        g_param: float=0.75
            For scale-free randomly-constructed networks, this determines the emphasis on the network's
            out degree distribution, and in practical terms, increases the scale-free character of the out-distribution
            of the graph. Note that 1 - beta - gamma must be greater than 0.0.

        delta_in: float=0.0
            A parameter that increases the complexity of the network core, leading to more nodes being involved in
            cycles.
        delta_out: float = 0.0
            A parameter that increases the complexity of the network core, leading to more nodes being involved in
            cycles.

        p_edge: float=0.2
            For randomly constructed binomial-type networks, this parameter determines the probability of forming
            an edge. As p_edge increases, the number of network edges increases drammatically.
        '''
        # Set some basic features of the network that may be overridden later:
        self.N_nodes = N_Nodes
        self._nodes_index = [i for i in range(self.N_nodes)]
        self.regular_node_inds = self._nodes_index
        self.nodes_list = [f'G{i}' for i in range(self.N_nodes)]

        # Initialize some object state variables:
        self._reduced_dims = False # Indicate that model is full dimensions
        self._solved_analytically = False # Indicate that the model does not have an analytical solution
        self._dcdt_vect_reduced_s = None # Initialize this to None
        self.process_params_s = []  # initialize this to be an empty list
        self.edge_types = None
        self.edges_index = None

        self.p_min = 1.0e-6 # small nonzero element for working with Hill versions
        self._push_away_from_zero = 0.1 # smallish constant to push initial guess of fsolve away from zero

    def build_network_from_edges(self, edges: list[tuple]):
        '''

        '''
        self._graph_type = GraphType.user
        self.edges_list = edges
        self.GG = nx.DiGraph(self.edges_list)
        self.N_edges = len(self.edges_list)
        self.nodes_list = sorted(self.GG.nodes())
        self.N_nodes = len(self.nodes_list)  # re-assign the node number in case specification was wrong

        self._make_node_edge_indices()

    def randomly_generate_special_network(self,
                                          b_param: float=0.15,
                                          g_param: float=0.8,
                                          delta_in: float=0.0,
                                          delta_out: float=0.0,
                                          p_edge: float=0.5,
                                          graph_type: GraphType = GraphType.scale_free
                                          ):
        '''
        Randomly generate a network with a scale-free or binomial degree distribution.

        Parameters
        ----------
        graph_type : GraphType = GraphType.scale_free
            The type of graph to generate in randomly-constructed networks.

        b_param : float = 0.20
            For scale-free randomly-constructed networks, this determines the amount of interconnectivity between
            the in and out degree distributions, and in practical terms, increases the number of cycles in the graph.
            Note that 1 - beta - gamma must be greater than 0.0.

        g_param : float=0.75
            For scale-free randomly-constructed networks, this determines the emphasis on the network's
            out degree distribution, and in practical terms, increases the scale-free character of the out-distribution
            of the graph. Note that 1 - beta - gamma must be greater than 0.0.

        delta_in : float=0.0
            A parameter that increases the complexity of the network core, leading to more nodes being involved in
            cycles.

        delta_out : float = 0.0
            A parameter that increases the complexity of the network core, leading to more nodes being involved in
            cycles.

        p_edge : float=0.2
            For randomly constructed binomial-type networks, this parameter determines the probability of forming
            an edge. As p_edge increases, the number of network edges increases dramatically.

        '''
        # Save all the construction parameters to file:
        self._beta = b_param
        self._gamma = g_param
        self._delta_in = delta_in
        self._delta_out = delta_out
        self._p_edge = p_edge
        self._graph_type = graph_type

        if graph_type is GraphType.scale_free:
            # generate a scale-free network with the supplied parameters...
            # The input scale-free probability is given as 1.0 minus beta and gamma, as all
            # three parameters must be constrained to add to 1.0:
            alpha = 1.0 - b_param - g_param

            # Generate a scale free graph with the settings:
            GGo = nx.scale_free_graph(self.N_nodes,
                                      alpha=alpha,
                                      beta=b_param,
                                      gamma=g_param,
                                      delta_in=delta_in,
                                      delta_out=delta_out,
                                      seed=None,
                                      initial_graph=None)

        elif graph_type is GraphType.random:
            # generate a random Erdos-Renyi network
            GGo = nx.erdos_renyi_graph(self.N_nodes,
                                       p_edge,
                                       seed=None,
                                       directed=True)

        else:
            raise Exception("Only scale-free and random (binomial) networks supported.")

        # obtain the unique edges only:
        self.edges_list = list(set(GGo.edges()))
        self.N_edges = len(self.edges_list)

        # As the scale_free_graph function can return duplicate edges, get rid of these
        # by re-defining the graph with the unique edges only:
        GG = nx.DiGraph(self.edges_list)

        self.nodes_list = np.arange(self.N_nodes).tolist()
        self.GG = GG
        self.edges_index = self.edges_list
        self.nodes_index = self.nodes_list

    #----Graph Building & Characterizing Methods ------
    def _make_node_edge_indices(self):
        '''
        Especially important for the case where node names are strings,
        this method creates numerical (int) indices for the nodes and
        stores them in a nodes_index. It does the same for nodes in edges,
        storing them in an edges_index object.

        '''
        # For this case the user may provide string names for
        # nodes, so we need to make numerical node and edge listings:
        self.nodes_index = []
        for nde_i, nn in enumerate(self.nodes_list):
            self.nodes_index.append(nde_i)

        self.edges_index = []
        for ei, (nni, nnj) in enumerate(self.edges_list):
            nde_i = self.nodes_list.index(nni)
            nde_j = self.nodes_list.index(nnj)
            self.edges_index.append((nde_i, nde_j))
        # self.nodes_list = np.arange(self.N_nodes)

    def characterize_graph(self):
        '''
        Perform a number of graph-theory style characterizations on the network to determine
        cycle number, analyze in- and out- degree distribution, and analyze hierarchy. Hierarchical
        structure analysis was from the work of Moutsinas, G. et al. Scientific Reports 11 (2021).

        '''
        # Indices of edges with selfloops:
        self.selfloop_edge_inds = [self.edges_list.index(ei) for ei in list(nx.selfloop_edges(self.GG))]

        # Degree analysis:
        self.in_degree_sequence = [deg_i for nde_i, deg_i in
                                   self.GG.in_degree(self.nodes_list)] # aligns with node order

        self.in_dmax = np.max(self.in_degree_sequence)


        self.out_degree_sequence = [deg_i for nde_i, deg_i in
                                    self.GG.out_degree(self.nodes_list)]  # aligns with node order

        # The outward flow of interaction at each node of the graph:
        self.node_divergence = np.asarray(self.out_degree_sequence) - np.asarray(self.in_degree_sequence)

        self.out_dmax = np.max(self.out_degree_sequence)
        self.in_dmax = np.max(self.in_degree_sequence)

        self.in_bins, self.in_degree_counts = np.unique(self.in_degree_sequence,
                                                        return_counts=True)
        self.out_bins, self.out_degree_counts = np.unique(self.out_degree_sequence,
                                                          return_counts=True)

        # Nodes sorted by number of out-degree edges:
        self.nodes_by_out_degree = np.flip(np.argsort(self.out_degree_sequence))

        self.nodes_by_in_degree = np.flip(np.argsort(self.in_degree_sequence))

        self.root_hub = self.nodes_by_out_degree[0]
        self.leaf_hub = self.nodes_by_out_degree[-1]

        # Number of cycles:
        self.graph_cycles = sorted(nx.simple_cycles(self.GG))
        self.N_cycles = len(self.graph_cycles)

        # Determine the nodes in the cycles:
        nodes_in_cycles = set()
        for nde_lst in self.graph_cycles:
            for nde_ni in nde_lst:
                nde_i = self.nodes_list.index(nde_ni)
                nodes_in_cycles.add(nde_i)

        self.nodes_in_cycles = list(nodes_in_cycles)
        self.nodes_acyclic = np.setdiff1d(self.nodes_index, nodes_in_cycles)


        # Graph structure characterization (from the amazing paper of Moutsinas, G. et al. Scientific Reports 11 (2021))
        a_out = list(self.GG.adjacency())

        # Adjacency matrix (outward connection directed)
        self.A_out = np.zeros((self.N_nodes, self.N_nodes))
        for nde_ni, nde_j_dict in a_out:
            nde_i = self.nodes_list.index(nde_ni) # get the index in case nodes are names
            for nde_nj, _ in nde_j_dict.items():
                nde_j = self.nodes_list.index(nde_nj) # get the index in case nodes are names
                self.A_out[nde_i, nde_j] += 1

        # Diagonalized in and out degree sequences for the nodes:
        D_in = np.diag(self.in_degree_sequence)
        D_out = np.diag(self.out_degree_sequence)

        if D_out.shape == self.A_out.shape:
            # Graph Laplacians for out and in distributions:
            L_out = D_out - self.A_out
            L_in = D_in - self.A_out

            # Moore-Penrose inverse of Graph Laplacians:
            L_in_inv = np.linalg.pinv(L_in.T)
            L_out_inv = np.linalg.pinv(L_out)

            # Grading of hierarchical level of nodes:
            # fwd hierachical levels grade vertices based on distance from source subgraphs
            self.fwd_hier_node_level = L_in_inv.dot(self.in_degree_sequence)
            # rev hierachical levels grade vertices based on distance from sink subgraphs
            self.rev_hier_node_level = L_out_inv.dot(self.out_degree_sequence)
            # overal hierachical levels of the graph (this is akin to a y-coordinate for each node of the network):
            self.hier_node_level = (1 / 2) * (self.fwd_hier_node_level - self.rev_hier_node_level)

            # Next, define a difference matrix for the network -- this calculates the difference between node i and j
            # as an edge parameter when it is dotted with a parameter defined on nodes:
            self.D_diff = np.zeros((self.N_edges, self.N_nodes))
            for ei, (nde_i, nde_j) in enumerate(self.edges_index):
                self.D_diff[ei, nde_i] = 1
                self.D_diff[ei, nde_j] = -1

            #Next calculate the forward and backward hierarchical differences:
            self.fwd_hier_diff = self.D_diff.dot(self.fwd_hier_node_level)
            self.rev_hier_diff = self.D_diff.dot(self.rev_hier_node_level)

            #The democracy coefficient parameter (measures how much the influencers of a graph are influenced
            #themselves):
            self.dem_coeff = 1 - np.mean(self.fwd_hier_diff)
            self.dem_coeff_rev = 1 - np.mean(self.rev_hier_diff)

            # And the hierarchical incoherence parameter (measures how much feedback there is):
            self.hier_incoherence = np.var(self.fwd_hier_diff)
            self.hier_incoherence_rev = np.var(self.rev_hier_diff)

            # A graph with high democracy coefficient and high incoherence has all verts with approximately the same
            # hierarchical level. The graph is influenced by a high percentage of vertices. In a graph with low democracy
            # coefficient and low incoherence, the graph is controlled by small percentage of vertices (maximally
            # hierarchical at zero demo coeff and zero incoherence).

        else:
            self.hier_node_level = np.zeros(self.N_nodes)
            self.hier_incoherence = 0.0
            self.dem_coeff = 0.0

    def get_paths_matrix(self) -> ndarray:
        '''
        Compute a matrix showing the number of paths from starting node to end node. Note that this
        matrix can be extraordinarily large in a complicated graph such as most binomial networks.

        Returns
        -------
        ndarray
            The paths matrix, which specifies the number of paths between one node index as row index and another
            node index as the column index.

        '''


        # What we want to show is that the nodes with the highest degree have the most connectivity to nodes in the network:
        # mn_i = 10 # index of the master node, organized according to nodes_by_out_degree
        paths_matrix = []
        for mn_i in range(len(self.nodes_index)):
            number_paths_to_i = []
            for i in range(len(self.nodes_index)):
                # print(f'paths between {mn_i} and {i}')
                try:
                    paths_i = sorted(nx.shortest_simple_paths(self.GG,
                                                              self.nodes_by_out_degree[mn_i],
                                                              self.nodes_by_out_degree[i]),
                                     reverse=True)
                except:
                    paths_i = []

                num_paths_i = len(paths_i)
                number_paths_to_i.append(num_paths_i)

            paths_matrix.append(number_paths_to_i)

        self.paths_matrix = np.asarray(paths_matrix)

        return self.paths_matrix

    def get_edge_types(self, p_acti: float=0.5, set_selfloops_acti: bool=True):
        '''
        Automatically generate an edge "type" vector for use in model building.
        Edge type specifies whether the edge is an activating or inhibiting
        relationship between the nodes. This routine randomly chooses a set of
        activating and inhibiting edge types for a model.

        Parameters
        ----------
        p_acti : float = 0.5
            The probability of an edge being an activation. Note that this value
            must be less than 1.0, and that the probability of an edge being an
            inhibitor becomes 1.0 - p_acti.

        set_selfloops_acti : bool = True
            Work shows that self-inhibition does not generate models with multistable
            states. Therefore, this edge-type assignment routine allows one to
            automatically set all self-loops to be activation interactions.

        '''

        p_inhi = 1.0 - p_acti

        edge_types_o = [EdgeType.A, EdgeType.I]
        edge_prob = [p_acti, p_inhi]
        edge_types = np.random.choice(edge_types_o, self.N_edges, p=edge_prob)

        if set_selfloops_acti: # if self-loops of the network are forced to be activators:
            edge_types[self.selfloop_edge_inds] = EdgeType.A

        return edge_types.tolist()

    def set_edge_types(self, edge_types: list|ndarray):
        '''
        Assign edge_types to the graph and create an edge function list used in analytical model building.

        Parameters
        ----------
        edge_types : list
            A list of edge type enumerations; one for each edge of the network.

        add_interactions : bool
            In a network, the interaction of two or more regulators at a node can be multiplicative
            (equivalent to an "And" condition) or additive (equivalent to an "or condition). This
            bool specifies whether multiple interactions should be additive (True) or multiplicative (False).
        '''
        self.edge_types = edge_types

        # assign the edge types to the graph in case we decide to save the network:
        edge_attr_dict = {}
        for ei, et in zip(self.edges_list, edge_types):
            edge_attr_dict[ei] = {"edge_type": et.name}

        nx.set_edge_attributes(self.GG, edge_attr_dict)

    def set_node_types(self, node_type_dict: dict|None = None, pure_gene_edges_only: bool = False):
        '''
        Assign node types to the graph.

        Parameters
        ----------
        node_types : list
            A list of node type enumerations for each node of the network.
        '''

        # Now that indices are set, give nodes a type attribute and classify node inds.
        # First, initialize a dictionary to collect all node indices by their node type:
        self.node_type_inds = {}
        for nt in NodeType:
            self.node_type_inds[nt.name] = []

        # Next, set all nodes to the gene type by default:
        node_types = [NodeType.gene for i in self.nodes_index]

        # If there is a supplied node dictionary, go through it and
        # override the default gene type with the user-specified type:
        if node_type_dict is not None:
            for ntag, ntype in node_type_dict.items():
                for nde_i, nde_n in enumerate(self.nodes_list):
                    if type(nde_n) is str:
                        if nde_n.startswith(ntag):
                            node_types[nde_i] = ntype
                    else:
                        if nde_n == ntag:
                            node_types[nde_i] = ntype

        # Set node types to the graph:
        self.node_types = node_types
        # Set node type as graph node attribute:
        node_attr_dict = {}
        for nde_i, nde_t in zip(self.nodes_list, node_types):
            node_attr_dict[nde_i] = {"node_type": nde_t.name}

        nx.set_node_attributes(self.GG, node_attr_dict)

        # Collect node indices by their type:
        for nde_i, nde_t in enumerate(self.node_types):
            self.node_type_inds[nde_t.name].append(nde_i)

        # Next, we need to distinguish edges based on their node type
        # to separate out some node type interactions from the regular
        # GRN-type interactions:
        if pure_gene_edges_only:  # if the user wants to consider only gene type nodes
            type_tags = [NodeType.gene.name]
        else:  # Otherwise include all nodes that can form regular interaction edges:
            type_tags = [NodeType.gene.name,
                         NodeType.signal.name,
                         NodeType.sensor.name,
                         NodeType.effector.name]

        self.regular_node_inds = []
        for tt in type_tags:
            self.regular_node_inds.extend(self.node_type_inds[tt])

        # aliases for convenience:
        # combine signals with factors as they have a similar 'setability' condition
        # from the outside
        # self.input_node_inds = self.node_type_inds[NodeType.signal.name] + self.node_type_inds[NodeType.factor.name]
        self.input_node_inds = ((np.asarray(self.in_degree_sequence) == 0).nonzero()[0]).tolist()
        self.sensor_node_inds = self.node_type_inds[NodeType.sensor.name]
        self.process_node_inds = self.node_type_inds[NodeType.process.name]
        self.effector_node_inds = ((np.asarray(self.out_degree_sequence) == 0).nonzero()[0]).tolist()
        self.noninput_node_inds = np.setdiff1d(self.nodes_index, self.input_node_inds).tolist()
        self.factor_node_inds = self.node_type_inds[NodeType.factor.name]

    def edges_from_path(self, path_nodes: list|ndarray) -> list:
        '''
        If specifying a path in terms of a set of nodes, this method
        returns the set of edges corresponding to the path.

        Parameters
        ----------
        path_nodes : list
            A list of nodes in the network over which the path is specified.

        Returns
        -------
        list
            The list of edges corresponding to the path.

        '''
        path_edges = []
        for i in range(len(path_nodes)):
            if i != len(path_nodes) - 1:
                ei = (path_nodes[i], path_nodes[i + 1])
                path_edges.append(ei)

        return path_edges

    def read_edge_info_from_graph(self):
        '''

        '''
        self.edges_list = []
        self.edge_types = []

        # get data stored on edge type key:
        edge_data = nx.get_edge_attributes(self.GG, "edge_type")

        for ei, et in edge_data.items():
            # append the edge to the list:
            self.edges_list.append(ei)

            if et == 'Activator':
                self.edge_types.append(EdgeType.A)
            elif et == 'Inhibitor':
                self.edge_types.append(EdgeType.I)
            elif et == 'Normal':
                self.edge_types.append(EdgeType.N)
            else:
                raise Exception("Edge type not found.")

        self.N_edges = len(self.edges_list)

    def read_node_info_from_graph(self):
        '''

        '''

        self.node_types = []

        # get data stored on edge type key:
        node_data = nx.get_node_attributes(self.GG, "node_type")

        for nde_i, nde_t in node_data.items():
            if nde_t == 'Gene':
                self.node_types.append(NodeType.gene)
            elif nde_t == 'Process':
                self.node_types.append(NodeType.process)
            elif nde_t == 'Sensor':
                self.node_types.append(NodeType.sensor)
            elif nde_t == 'Effector':
                self.node_types.append(NodeType.effector)
            elif nde_t == 'Root Hub':
                self.node_types.append(NodeType.core)
            elif nde_t == 'Path':
                self.node_types.append(NodeType.path)
            else:
                raise Exception("Node type not found.")

    #----Methods for Analytical Models--------
    def reduce_model_dimensions(self):
        '''
        Using analytical methods, attempt to reduce the multidimensional
        network equations to as few equations as possible.

        '''

        # Solve the nonlinear system as well as is possible:
        nosol = False

        # We want to restrict this model reduction to homogeneous networks that
        # only contain genes and signals.
        if len(self.process_node_inds) or self._inter_fun_type is InterFuncType.logistic:
            nosol = True # Immediately flag the sysem as unsolvable

        else:

            try:
                sol_csetoo = sp.nonlinsolve(self._dcdt_vect_s, self._c_vect_s)
                # Clean up the sympy container for the solutions:
                sol_cseto = list(list(sol_csetoo)[0])

                if len(sol_cseto):

                    c_master_i = []  # the indices of concentrations involved in the master equations (the reduced dims)
                    sol_cset = {}  # A dictionary of auxillary solutions (plug and play)
                    for i, c_eq in enumerate(sol_cseto):
                        if c_eq in self._c_vect_s:  # If it's a non-solution for the term, append it as a non-reduced conc.
                            c_master_i.append(self._c_vect_s.index(c_eq))
                        else:  # Otherwise append the plug-and-play solution set:
                            sol_cset[self._c_vect_s[i]] = c_eq

                    master_eq_list = []  # master equations to be numerically optimized (reduced dimension network equations)
                    c_vect_reduced = []  # concentrations involved in the master equations

                    if len(c_master_i):
                        for ii in c_master_i:
                            # substitute in the expressions in terms of master concentrations to form the master equations:
                            ci_solve_eq = self._dcdt_vect_s[ii].subs([(k, v) for k, v in sol_cset.items()])
                            master_eq_list.append(ci_solve_eq)
                            c_vect_reduced.append(self._c_vect_s[ii])

                    else:  # if there's nothing in c_master_i but there are solutions in sol_cseto, then it's been fully solved:
                        print("The system has been fully solved by analytical methods!")
                        self._solved_analytically = True

                else:
                    nosol = True

            except:
                nosol = True

        # Results:
        if nosol is True:
            self._reduced_dims = False
            print("Unable to reduce equations!")
            # Set all reduced system attributes to None:
            self._dcdt_vect_reduced_s = None
            self._c_vect_reduced_s = None
            self.c_master_i = None
            self.c_remainder_i = None
            self.c_vect_remainder_s = None
            # This is the dictionary of remaining concentrations that are in terms of the reduced concentrations,
            # such that when the master equation set is solved, the results are plugged into the equations in this
            # dictionary to obtain solutions for the whole network
            self.sol_cset_s = None

            self.signal_reduced_inds = None
            self.nonsignal_reduced_inds = None

        else: # If we have solutions, proceed:
            self._reduced_dims = True

            if self._solved_analytically is False:
                # New equation list to be numerically optimized (should be significantly reduced dimensions):
                # Note: this vector is no longer the change rate vector; its now solving for concentrations
                # where the original rate change vector is zero:
                self._dcdt_vect_reduced_s = sp.Matrix(master_eq_list)
                # This is the concentration vector that contains the reduced equation concentration variables:
                self._c_vect_reduced_s = c_vect_reduced
                self.c_master_i = c_master_i # indices of concentrations that are being numerically optimized
                self.c_remainder_i = np.setdiff1d(self.nodes_index, self.c_master_i) # remaining conc indices
                self.c_vect_remainder_s = np.asarray(self._c_vect_s)[self.c_remainder_i].tolist() # remaining concs

                # This is the dictionary of remaining concentrations that are in terms of the reduced concentrations,
                # such that when the master equation set is solved, the results are plugged into the equations in this
                # dictionary to obtain solutions for the whole network:
                self.sol_cset_s = sol_cset

                # Create a set of signal node indices to the reduced c_vect array:
                self.signal_reduced_inds = []
                for si in self.input_node_inds:
                    if si in self.c_master_i:
                        self.signal_reduced_inds.append(self.c_master_i.index(si))

                self.nonsignal_reduced_inds = np.setdiff1d(np.arange(len(self.c_master_i)),
                                                           self.signal_reduced_inds)

            else:
                # Set most reduced system attributes to None:
                self._dcdt_vect_reduced_s = None
                self._c_vect_reduced_s = None
                self.c_master_i = None
                self.c_remainder_i = None
                self.c_vect_remainder_s = None
                self.signal_reduced_inds = None
                self.nonsignal_reduced_inds = None


                # This is the dictionary of remaining concentrations that are in terms of the reduced concentrations,
                # such that when the master equation set is solved, the results are plugged into the equations in this
                # dictionary to obtain solutions for the whole network:
                self.sol_cset_s = sol_cset

                # The sol_cset exists and can be lambdified for full solutions. Here we lambdify it without the c_vect:
                if len(self.process_node_inds):
                    lambda_params_r = [self.d_vect_s,
                                       self.B_vect_s,
                                       self.n_vect_s,
                                       self.extra_params_s]

                else:
                    lambda_params_r = [self.d_vect_s,
                                       self.B_vect_s,
                                       self.n_vect_s]

                self.sol_cset_f = {}
                for ci, eqci in self.sol_cset_s.items():
                    self.sol_cset_f[ci.indices[0]] = sp.lambdify(lambda_params_r, eqci)

    #----Time Dynamics Methods----------------
    def pulses(self,
               tvect: list|ndarray,
               t_on: float|int,
               t_off: float|int,
               c_base: float | int = 1.0,
               c_active: float | int = 0.0,
               ):
        '''

        '''
        itop = (tvect >= t_on).nonzero()[0]
        ibot = (tvect <= t_off).nonzero()[0]

        ipulse = np.intersect1d(ibot, itop)

        pulse_sig = c_active * np.ones(len(tvect))

        pulse_sig[ipulse] = c_base

        return pulse_sig

    def get_interval_inds(self,
                           tvect: ndarray,
                           t_on: float,
                           t_off: float,
                           t_wait: float = 0.0
                           ):
        '''
        Returns indices specifying an interval from the
        supplied vector tvect spanning between vector
        values t_on and t_off.

        Parameters
        ----------
        t_vect : ndarray
            The vector to pull the interval from.

        t_on : float
            The first value in the vector defining the interval start.

        t_off : float
            The value in the vector defining the interval end.

        t_wait: float
            The amount of time to push up the start of the interval from
            t_on (which is useful if you're waiting for the system to get
            back to steady-state).

        '''
        itop = (tvect >= t_on + t_wait).nonzero()[0]
        ibot = (tvect <= t_off).nonzero()[0]

        ipulse_inds = np.intersect1d(ibot, itop)

        return ipulse_inds

    def make_pulsed_signals_matrix(self,
                                   tvect: list|ndarray,
                                   sig_inds: list|ndarray,
                                   sig_times: list|ndarray,
                                   sig_mag: list|ndarray):
        '''

        '''
        Nt = len(tvect)

        c_signals = np.zeros((Nt, self.N_nodes))  # Initialize matrix holding the signal sequences

        for si, (ts, te), (smin, smax) in zip(sig_inds, sig_times, sig_mag):
            c_signals[:, si] += self.pulses(tvect,
                                            ts,
                                            te,
                                            c_base=smax,
                                            c_active=smin
                                            )

        return c_signals

    def get_all_intervals(self,
                          tvect: ndarray,
                          sig_times: list | ndarray,
                          t_wait: float = 0.0,
                          add_end_intervals: bool = True
                          ):
        '''

        '''
        intervals_set = set()  # Initialize set to hold the interval indices

        # sig_times = sorted(sig_times) # sort the time tuples by their start time
        sig_times.sort(key=lambda x: x[0])  # sort the time tuples by their start time

        for ts, te in sig_times:
            inti = self.get_interval_inds(tvect,
                                          ts,
                                          te,
                                          t_wait=t_wait
                                          )

            if len(inti):
                intervals_set.add((inti[0], inti[-1]))

        if add_end_intervals:
            # Add a start interval
            intis = self.get_interval_inds(tvect,
                                           tvect[0],
                                           sig_times[0][0],
                                           t_wait=t_wait
                                           )
            if len(intis):
                intervals_set.add((intis[0], intis[-1]))

            # Add an end interval:
            intie = self.get_interval_inds(tvect,
                                           sig_times[-1][1],
                                           tvect[-1],
                                           t_wait=t_wait
                                           )
            if len(intie):
                intervals_set.add((intie[0], intie[-1]))

        intervals_list = list(intervals_set)
        intervals_list.sort(key=lambda x: x[0])  # sort the time tuples by their start time

        return intervals_list

    def make_time_vects(self,
                        tend: float,
                        dt: float,
                        dt_samp: float|None = None,):
        '''

        '''
        Nt = int(tend/dt)
        tvect = np.linspace(0.0, tend, Nt)

        # sampling compression
        if dt_samp is not None:
            sampr = int(dt_samp / dt)
            tvectr = tvect[0::sampr]
        else:
            tvectr = tvect

        # make a time-step update vector so we can update any sensors as
        # an absolute reading (dt = 1.0) while treating the kinetics of the
        # other node types:
        # dtv = dt * np.ones(self._N_nodes)
        # dtv[self.sensor_node_inds] = 1.0

        return tvect, tvectr

    @abstractmethod
    def run_time_sim(self,
                     tvect: ndarray|list,
                     tvectr: ndarray|list,
                     cvecti: ndarray|list,
                     sig_inds: ndarray|list|None = None,
                     sig_vals: list | ndarray | None = None,
                     constrained_inds: list | None = None,
                     constrained_vals: list | None = None,
                     d_base: float = 1.0,
                     n_base: float = 15.0,
                     beta_base: float = 0.25
                     ):
        '''

        '''
        pass

    #----Plots and Data Export----------------
    def save_network(self, filename: str):
        '''
        Write a network, including edge types, to a saved file.

        '''
        nx.write_gml(self.GG, filename)

    def save_network_image(self, save_filename: str, use_dot_layout: bool=False):
        '''
        Uses pygraphviz to create a nice plot of the network model.

        '''
        G_plt = pgv.AGraph(strict=False,
                           splines=True,
                           directed=True,
                           randkdir='TB',
                           nodesep=0.1,
                           ranksep=0.3,
                           dpi=300)

        for nde_i in self.nodes_list:
            G_plt.add_node(nde_i,
                           style='filled',
                           fillcolor='LightCyan',
                           color='Black',
                           shape='ellipse',
                           fontcolor='Black',
                           # fontname=net_font_name,
                           fontsize=12)

        for (ei, ej), etype in zip(self.edges_list, self.edge_types):
            if etype is EdgeType.A:
                G_plt.add_edge(ei, ej, arrowhead='dot', color='blue', penwidth=2.0)
            elif etype is EdgeType.I:
                G_plt.add_edge(ei, ej, arrowhead='tee', color='red', penwidth=2.0)
            else:
                G_plt.add_edge(ei, ej, arrowhead='normal', color='black', penwidth=2.0)

        if use_dot_layout:
            G_plt.layout(prog="dot")
        else:
            G_plt.layout()

        G_plt.draw(save_filename)

    def plot_degree_distributions(self):
        '''

        '''
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        ax[0].bar(self.in_bins, self.in_degree_counts)
        ax[0].set_xlabel('Node degree')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('In-Degree Distribution')
        ax[1].bar(self.out_bins, self.out_degree_counts)
        ax[1].set_xlabel('Node degree')
        # ax[1].set_ylabel('Counts')
        ax[1].set_title('Out-Degree Distribution')

        return fig, ax

    def plot_sols_array(self, solsM: ndarray, figsave: str | None = None, cmap: str | None =None):
        '''

        '''

        if cmap is None:
            cmap = 'magma'

        state_labels = [f'State {i +1}' for i in range(solsM.shape[1])]
        gene_labels = np.asarray(self.nodes_list)[self.regular_node_inds]
        fig, ax = plt.subplots()
        im = ax.imshow(solsM[self.regular_node_inds, :], cmap=cmap)
        # plt.colorbar(label='Expression Level')
        ax.set_xticks(np.arange(len(state_labels)), labels=state_labels)
        ax.set_yticks(np.arange(len(gene_labels)), labels=gene_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label='Expression Level')

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format='png')

        return fig, ax

    def plot_pixel_matrix(self,
                          solsM: ndarray,
                          gene_labels: list|ndarray,
                          figsave: str|None = None,
                          cmap: str|None =None,
                          cbar_label: str=''):
        '''
        Plot a correlation or adjacency matrix for a subset of genes.

        '''

        if cmap is None:
            cmap = 'magma'

        fig, ax = plt.subplots()
        im = ax.imshow(solsM, cmap=cmap)
        ax.set_xticks(np.arange(len(gene_labels)), labels=gene_labels)
        ax.set_yticks(np.arange(len(gene_labels)), labels=gene_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label=cbar_label)

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format='png')

        return fig, ax


    def save_model_equations(self,
                             save_eqn_image: str,
                             save_reduced_eqn_image: str|None = None,
                             save_eqn_csv: str|None = None,
                             substitute_node_labels: bool = True
                             ):
        '''
        Save images of the model equations, as well as a csv file that has
        model equations written in LaTeX format.

        Parameters
        -----------
        save_eqn_image : str
            The path and filename to save the main model equations as an image.

        save_reduced_eqn_image : str|None = None
            The path and filename to save the reduced main model equations as an image (if model is reduced).

        save_eqn_csv : str|None = None
            The path and filename to save the main and reduced model equations as LaTex in a csv file.

        '''

        if substitute_node_labels:
            subs_list = []
            for pi, nde_lab in zip(self._c_vect_s, self.nodes_list):
                subs_list.append((pi, nde_lab))

            for ei, (nij, bij) in enumerate(zip(self._n_vect_s, self._beta_vect_s)):
                subs_list.append((bij, f'beta_{ei}'))
                subs_list.append((nij, f'n_{ei}'))

            _dcdt_vect_s = list(sp.Matrix(self._dcdt_vect_s).subs(subs_list))
            _c_vect_s = list(sp.Matrix(self._c_vect_s).subs(subs_list))

            if self._dcdt_vect_reduced_s is not None:
                _dcdt_vect_reduced_s = list(sp.Matrix(self._dcdt_vect_reduced_s).subs(subs_list))
                _c_vect_reduced_s = list(sp.Matrix(self._c_vect_reduced_s).subs(subs_list))

        else:
            _dcdt_vect_s = self._dcdt_vect_s
            _c_vect_s = self._c_vect_s

            if self._dcdt_vect_reduced_s is not None:
                _dcdt_vect_reduced_s = self._dcdt_vect_reduced_s
                _c_vect_reduced_s = self._c_vect_reduced_s


        t_s = sp.symbols('t')
        c_change = sp.Matrix([sp.Derivative(ci, t_s) for ci in _c_vect_s])
        eqn_net = sp.Eq(c_change, sp.Matrix(_dcdt_vect_s))

        sp.preview(eqn_net,
                   viewer='file',
                   filename=save_eqn_image,
                   euler=False,
                   dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600", "-bg", "Transparent"])

        # Save the equations for the graph to a file:
        header = ['Concentrations', 'Change Vector']
        eqns_to_write = [[sp.latex(_c_vect_s), sp.latex(_dcdt_vect_s)]]

        if self._dcdt_vect_reduced_s is not None and save_reduced_eqn_image is not None:
            c_change_reduced = sp.Matrix([sp.Derivative(ci, t_s) for ci in _c_vect_reduced_s])
            eqn_net_reduced = sp.Eq(c_change_reduced, _dcdt_vect_reduced_s)

            sp.preview(eqn_net_reduced,
                       viewer='file',
                       filename=save_reduced_eqn_image,
                       euler=False,
                       dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600", "-bg", "Transparent"])

            eqns_to_write.append(sp.latex(_c_vect_reduced_s))
            eqns_to_write.append(sp.latex(_dcdt_vect_reduced_s))
            header.extend(['Reduced Concentations', 'Reduced Change Vector'])

        if save_eqn_csv is not None:
            with open(save_eqn_csv, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # 2. create a csvwriter object
                csvwriter.writerow(header)  # 4. write the header
                csvwriter.writerows(eqns_to_write)  # 5. write the rest of the data

    #----Heterogeneous Networks Methods -------
    def osmotic_process(self,
                        c_effector: Symbol | Indexed,
                        c_process: Symbol | Indexed,
                        c_factor: Symbol | Indexed,
                        c_channel: Symbol | Indexed | None = None):
        '''
        This method generates the symbolic and numerical equations required to define an
        osmotic process, whereby differences in osmolyte concentrations on either side of
        the membrane generate circumferential strains in a cell due to volume shrinkage
        (case where c_effector < c_factor; c_process will be negative reflecting shrinkage
        strain) or volume expansion (case where c_effector > c_factor; c_process will be
        positive reflecting expansive strain). The functions return the strain rate for the
        system, which is substituted into the system change vector.  In these expresions, all
        parameters are normalized so that their values lay between 0 and 1 under most
        circumstances.

        gmod : GeneNetworkModel
            An instance of gene network model in which to add the osmotic process equations.

        c_effector : Symbol
            Symbolic concentration from a node in the GRN network that represents the moles of
            osmolyte inside the cell. This should correspond with an 'Effector' type node.

        c_process : Symbol
            Symbolic concentration from a node in the GRN network that represents the circumferential
            strain of the cell membrane. Note when c_process < 0, the cell has shrunk and when
            c_process is > 0, the cell has expanded from its equilibrium volume state.
            This should correspond with a 'Process' type node.

        c_factor : Symbol
            Symbolic concentration from a node in the GRN network that represents the moles of
            osmolyte outside the cell. This should correspond with a 'Factor' type node.

        c_channel : Symbol|Indexed|None = None
            Optional parameter allowing for use of a second effector, which controls the
            expression of water channels.

        '''
        # FIXME: we need to put in the c_factors here for the environmental osmolytes

        # Defining analytic equations for an osmotic cell volume change process:
        A_s, R_s, T_s, ni_s, m_s, V_s, Vc_s, dm_s, mu_s, Y_s, r_s = sp.symbols('A, R, T, ni, m, V, V_c, d_m, mu, Y, r',
                                                                               real=True)
        # Normalized parameters:
        Ap_s, mp_s, Ac_s, nc_s, epsilon_s = sp.symbols('A_p, m_p, A_c, n_c, epsilon', real=True)
        if c_channel is not None:
            # if c_channel is not None, then redefine the Normalized channel parameter so it
            # corresponds with the supplied symbol:
            Ap_s = c_channel

        dVdt_0_s = A_s ** 2 * R_s * T_s * (ni_s - m_s * V_s) / (8 * dm_s * mu_s * V_s)
        dVdt_1_s = (A_s ** 2 / (8 * dm_s * mu_s)) * (
                R_s * T_s * ((ni_s / V_s) - m_s) - sp.Rational(4, 3) * ((Y_s * dm_s * (V_s - Vc_s) / (r_s * Vc_s))))

        # Rate of change of Vp with respect to time for Vp < 1.0 is:
        dVpdt_0_s = (dVdt_0_s.subs(
            [(V_s, (c_process + 1) * Vc_s),
             (A_s, Ap_s * Ac_s),
             (ni_s, nc_s * c_effector), (m_s, c_factor * mp_s)]) / Vc_s).simplify()

        # Rate of change of Vp with respect to time for Vp >= 1.0
        dVpdt_1_s = (dVdt_1_s.subs(
            [(V_s, (c_process + 1) * Vc_s), (A_s, Ap_s * Ac_s), (ni_s, nc_s * c_effector),
             (m_s, c_factor * mp_s)]) / Vc_s).simplify()

        # Volume change rates (which are the input into the sensor node) are:
        dEdt_0_s = dVpdt_0_s
        dEdt_1_s = dVpdt_1_s

        # Piecewise function that defines this normalized-parameter osmotic cell volume change problem
        # as a strain rate:
        self.dEdt_s = sp.Piecewise((dEdt_0_s, c_process < 0.0), (dEdt_1_s, True))

        # Go ahead and initialize some parameters for this process function: # FIXME these need to be
        # made easier to input, vary and change:
        # self.m_f = 0.8  # Normalized environmental osmolyte concentration (high)
        self.A_f = 1.0  # Normalized water/glycerol channel area
        self.R_f = 8.3145  # Ideal Gas constant
        self.T_f = 310.0  # System temperature in K
        self.Y_f = 100e6  # Young's modulus for membrane or wall
        self.dm_f = 1.0e-6  # membrane or wall thickness
        self.mu_f = 1.0e-3  # water viscosity
        self.r_f = 10.0e-6  # Undeformed cell radius
        self.Vc_f = 2 * np.pi * self.r_f ** 2  # undeformed cell volume
        self.nc_f = 1000.0 * self.Vc_f  # osmolyte moles in the cell (near max)
        self.mc_f = 1000.0  # osmolyte concentration in the environment (near max)
        self.Ac_f = 0.12e6 * np.pi * 1.0e-9 ** 2  # normalizing total water channel area (near max)

        self.Vp_min = 0.2  # minimum relative volume that can be achieved
        self.Vp_max = 2.0  # maximum relative volume that can be achieved

        # As strains (which is the format of the process parameter):
        self.strain_min = self.Vp_min - 1.0
        self.strain_max = self.Vp_max - 1.0


        # symbolic parameters for the dV/dt process (these must be augmented onto the GRN parameters
        # when lambdifying):

        if c_channel is None:
            self.process_params_s = (Ap_s, Vc_s, nc_s, mp_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s)
            self.dEdt_f = sp.lambdify([c_process, c_effector, c_factor, self.process_params_s], self.dEdt_s)

            # Numeric parameters for the dV/dt process (these must be augmented onto the GRN parameters
            # when using numerical network equations):
            self.process_params_f = (
                self.A_f,
                self.Vc_f,
                self.nc_f,
                self.mc_f,
                self.Ac_f,
                self.R_f,
                self.T_f,
                self.Y_f,
                self.dm_f,
                self.mu_f,
                self.r_f)

        else:
            self.process_params_s = (Vc_s, nc_s, mp_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s)
            self.dEdt_f = sp.lambdify([c_process, c_effector, c_channel, c_factor, self.process_params_s], self.dEdt_s)

            # Numeric parameters for the dV/dt process (these must be augmented onto the GRN parameters
            # when using numerical network equations):
            self.process_params_f = (
                self.Vc_f,
                self.nc_f,
                self.mc_f,
                self.Ac_f,
                self.R_f,
                self.T_f,
                self.Y_f,
                self.dm_f,
                self.mu_f,
                self.r_f)



