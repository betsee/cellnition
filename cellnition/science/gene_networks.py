#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds the model network as a symbolic graph, has attributes to
analyze the network, and has the ability to build an analytic (symbolic) model
that can be used to study the network as a continuous dynamic system.

Note on model parameterization:
For a general regulatory network, one can say the rate of change of agent a_i is:
d a_i/dt = r_max*sum(f(a_j)) - a_i*d_max
Where d_max is maximum rate of decay, r_max is maximum rate of growth, and f(a_j) is
an interaction function detailing how ajent a_j influences the growth of a_i.

Here we use normalized agent variables: c_i = a_i/alpha with alpha = (r_i/d_i).
We use the substitution, a_i = c_i*alpha for all entities in the network rate equations.
Then we note that if we're using Hill equations, then for each edge with index ei and
node index i acting on node j we can define an additional parameter,
beta_ei = r_i/(K_ei*d_i) where K_ei is the Hill coefficient for the edge interaction, and
r_i and d_i are the maximum rate of growth and decay (respectively) for node i acting on j
via edge ei.

The result is an equation, which at steady-state is only dependent on the parameters beta_ei and
the Hill exponent n_ei. In kinetics, the node d_i multiplies through the equation to define a
relative rate of change, however, in steady-state searches this d_i multiplies out (assuming d_i != 0).
'''
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from scipy.optimize import minimize, fsolve
from scipy.signal import square
import networkx as nx
from networkx import DiGraph
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
from cellnition.science.interaction_functions import f_acti_s, f_inhi_s, f_neut_s, f_logi_s
import pygraphviz as pgv

# TODO: Add in stochasticity
# TODO: In time sims, remember to deal separately with any sensor nodes
# TODO: In edgetype and param space searches, remember that the network may now have a greatly
# reduced set of parameters to work with and fit.

class GeneNetworkModel(object):
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

            dcdt_vect_reduced_s = None

        self.nodes_in_cycles = list(nodes_in_cycles)
        self.nodes_acyclic = np.setdiff1d(self.nodes_index, nodes_in_cycles)

                    self.hier_node_level = np.zeros(self.N_nodes)
            self.hier_incoherence = 0.0
            self.dem_coeff = 0.0

    '''

    def __init__(self,
                 N_nodes: int,
                 edges: list|ndarray|None = None,
                 graph_type: GraphType = GraphType.scale_free,
                 beta: float = 0.20,
                 gamma: float=0.75,
                 delta_in: float=0.0,
                 delta_out: float = 0.0,
                 p_edge: float=0.2
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

        beta: float = 0.20
            For scale-free randomly-constructed networks, this determines the amount of interconnectivity between
            the in and out degree distributions, and in practical terms, increases the number of cycles in the graph.
            Note that 1 - beta - gamma must be greater than 0.0.

        gamma: float=0.75
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
        self.N_nodes = N_nodes # number of nodes in the network (as defined by user initially)
        self._graph_type = graph_type

        # Depending on whether edges are supplied by user, generate
        # a graph:

        if edges is None:
            self.generate_network(beta=beta,
                                  gamma=gamma,
                                  graph_type=graph_type,
                                  delta_in=delta_in,
                                  delta_out=delta_out,
                                  p_edge=p_edge)

            self.edges_index = self.edges_list
            self.nodes_index = self.nodes_list

        else:
            self.edges_list = edges
            self.GG = nx.DiGraph(self.edges_list)
            self.N_edges = len(self.edges_list)
            self.nodes_list = sorted(self.GG.nodes())
            self.N_nodes = len(self.nodes_list) # re-assign the node number in case specification was wrong

            self._make_node_edge_indices()

        # Calculate key characteristics of the graph
        self.characterize_graph()

        # Initialize some object state variables:
        self._reduced_dims = False # Indicate that model is full dimensions
        self._solved_analytically = False # Indicate that the model does not have an analytical solution
        self.dcdt_vect_reduced_s = None # Initialize this to None

    def generate_network(self,
                         beta: float=0.15,
                         gamma: float=0.8,
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

        beta : float = 0.20
            For scale-free randomly-constructed networks, this determines the amount of interconnectivity between
            the in and out degree distributions, and in practical terms, increases the number of cycles in the graph.
            Note that 1 - beta - gamma must be greater than 0.0.

        gamma : float=0.75
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


        if graph_type is GraphType.scale_free:
            # generate a scale-free network with the supplied parameters...
            # The input scale-free probability is given as 1.0 minus beta and gamma, as all
            # three parameters must be constrained to add to 1.0:
            alpha = 1.0 - beta - gamma

            # Generate a scale free graph with the settings:
            GGo = nx.scale_free_graph(self.N_nodes,
                                      alpha=alpha,
                                      beta=beta,
                                      gamma=gamma,
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

        return edge_types

    def set_edge_types(self, edge_types: list|ndarray, add_interactions: bool):
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
            edge_attr_dict[ei] = {"edge_type": et.value}

        nx.set_edge_attributes(self.GG, edge_attr_dict)

        self.edge_funcs = []
        self.add_edge_interaction_bools = []
        self.growth_interaction_bools = []

        for et in self.edge_types:
            if et is EdgeType.A:
                self.edge_funcs.append(f_acti_s)
                self.add_edge_interaction_bools.append(add_interactions)
                self.growth_interaction_bools.append(True) # interact with growth component of reaction
            elif et is EdgeType.I:
                self.edge_funcs.append(f_inhi_s)
                self.add_edge_interaction_bools.append(add_interactions)
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction
            elif et is EdgeType.N:
                self.edge_funcs.append(f_neut_s)
                self.add_edge_interaction_bools.append(False)
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction
            elif et is EdgeType.As: # else if it's As for a sensor activation interaction, then:
                # self.edge_funcs.append(self.f_acti_s) # activate
                # self.add_edge_interaction_bools.append(True) # add
                # self.growth_interaction_bools.append(True)  # interact with growth component of reaction
                self.edge_funcs.append(f_inhi_s) # inhibit
                self.add_edge_interaction_bools.append(False) # multiply
                self.growth_interaction_bools.append(False)  # interact with decay component of reaction
            else:
                self.edge_funcs.append(f_inhi_s) # inhibit
                self.add_edge_interaction_bools.append(False) # multiply
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction

    def set_node_types(self, node_types: list|ndarray):
        '''
        Assign node types to the graph.

        Parameters
        ----------
        node_types : list
            A list of node type enumerations for each node of the network.
        '''
        self.node_types = node_types
        # Set node type as graph node attribute:
        node_attr_dict = {}
        for nde_i, nde_t in zip(self.nodes_index, node_types):
            node_attr_dict[nde_i] = {"node_type": nde_t.value}

        nx.set_node_attributes(self.GG, node_attr_dict)

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

    def build_analytical_model(self,
                               prob_acti: float=0.5,
                               edge_types: list|ndarray|None=None,
                               node_type_dict: dict|None=None,
                               add_interactions: bool=True,
                               pure_gene_edges_only: bool=False):
        '''
        Using the network to supply structure, this method constructs an
        analytical (symbolic) model of the regulatory network as a dynamic
        system.

        prob_acti : float = 0.5
            The probability of an edge being an activation. Note that this value
            must be less than 1.0, and that the probability of an edge being an
            inhibitor becomes 1.0 - p_acti.

        edge_types : list|None
            A list of edge type enumerations; one for each edge of the network. If edge_types
            is not specified, then they will be randomly generated, with all self-loops set to be
            activations.

        node_type_dict : dict|None
            Dictionary that allows for specification of different node types. If node names are
            strings, the node dict can be used to assign a node type based on the first letter
            of the node name. For example, if nodes labeled 'S0', 'S1', 'S2' are desired to be
            signal-type nodes, then the node dictionary would specify {'S': NodeType.signal}.
            Nodes that are numerically named must be individually specified by their number in the
            dictionary. If node_type_dict is None, then all nodes become NodeType.Gene.

        add_interactions : bool = False
            In a network, the interaction of two or more regulators at a node can be multiplicative
            (equivalent to an "And" condition) or additive (equivalent to an "or condition). This
            bool specifies whether multiple interactions should be additive (True) or multiplicative (False).

        pure_gene_edges_only : bool = False
            In a network with mixed node types, setting this to true will ensure that only edges
            that connect two nodes that are both NodeType.gene will be included in the model. If
            it's set to "False", then NodeType.gene, NodeType.sensor, and NodeType.effector are all
            included as valid edges.

        '''

        self._reduced_dims = False # always build models in full dimensions

        if edge_types is None:
            self.edge_types = self.get_edge_types(p_acti=prob_acti)

        else:
            self.edge_types = edge_types

        self.set_edge_types(self.edge_types, add_interactions)

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
        self.set_node_types(node_types)

        # Collect node indices by their type:
        for nde_i, nde_t in enumerate(self.node_types):
            self.node_type_inds[nde_t.name].append(nde_i)

        # Next, we need to distinguish edges based on their node type
        # to separate out some node type interactions from the regular
        # GRN-type interactions:
        if pure_gene_edges_only: # if the user wants to consider only gene type nodes
            type_tags = [NodeType.gene.name]
        else: # Otherwise include all nodes that can form regular interaction edges:
            type_tags = [NodeType.gene.name,
                         NodeType.signal.name,
                         NodeType.sensor.name,
                         NodeType.effector.name]

        self.regular_node_inds = []
        for tt in type_tags:
            self.regular_node_inds += self.node_type_inds[tt]

        # aliases for convenience:
        # combine signals with factors as they have a similar 'setability' condition
        # from the outside
        self.signal_node_inds = self.node_type_inds[NodeType.signal.name] + self.node_type_inds[NodeType.factor.name]
        self.sensor_node_inds = self.node_type_inds[NodeType.sensor.name]
        self.process_node_inds = self.node_type_inds[NodeType.process.name]
        self.nonsignal_node_inds = np.setdiff1d(self.nodes_index, self.signal_node_inds)

        # Next we want to distinguish a subset of edges that connect only "regular nodes":
        self.regular_edges_index = []
        for ei, (nde_i, nde_j) in enumerate(self.edges_index):
            if nde_i in self.regular_node_inds and nde_j in self.regular_node_inds:
                self.regular_edges_index.append((nde_i, nde_j))

        # Next, create a process dict that organizes features required for
        # constructing the heterogenous process:
        self.processes_list = []
        if len(self.process_node_inds):
            for pi in self.process_node_inds:
                process_dict = {NodeType.process.name: pi} # store the process node
                # Allow for multiple effectors, sensors and factors:
                process_dict[NodeType.effector.name] = []
                process_dict[NodeType.sensor.name] = []
                process_dict[NodeType.factor.name] = []
                # set through all edges:
                for ei, (nde_i, nde_j) in enumerate(self.edges_index):
                    # Find an edge that connects to this process
                    if pi == nde_i:
                        process_dict[self.node_types[nde_j].name].append(nde_j)
                    elif pi == nde_j:
                        process_dict[self.node_types[nde_i].name].append(nde_i)

                self.processes_list.append(process_dict)

        B_s = sp.IndexedBase('beta') # Symbolic base Hill 'beta parameter' edge interaction
        n_s = sp.IndexedBase('n') # Symbolic base Hill exponent for each edge interaction
        d_max_s = sp.IndexedBase('d_max') # Symbolic base decay rate for each node

        # These are needed for lambdification of analytical models:
        self.B_vect_s = [B_s[i] for i in range(self.N_edges)]
        self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
        self.d_vect_s = [d_max_s[i] for i in self.nodes_index]

        if type(self.nodes_list[0]) is str:
            # If nodes have string names, let the math reflect these:
            self.c_vect_s = [sp.symbols(ndi) for ndi in self.nodes_list]
        else:
            # Otherwise, if nodes are just numbered, let it be a generic 'c' symbol
            # to node number:
            c_s = sp.IndexedBase('c')
            self.c_vect_s = [c_s[i] for i in self.nodes_index]

        # If there are sensors in the network, described by logistic function
        # they have their own parameters:
        if len(self.sensor_node_inds):
            k_s = sp.IndexedBase('k') # steepness of sensor reaction
            co_s = sp.IndexedBase('co') # centre of sensor reaction
            self.sensor_params_s = {}
            for ii, sens_i in enumerate(self.sensor_node_inds):
                self.sensor_params_s[sens_i] = (co_s[ii], k_s[ii])
        else:
            self.sensor_params_s = {}

        self.process_params_s = [] # initialize this to be an empty list

        efunc_add_growthterm_vect = [[] for i in self.nodes_index]
        efunc_mult_growthterm_vect = [[] for i in self.nodes_index]
        efunc_mult_decayterm_vect = [[] for i in self.nodes_index]

        for ei, ((i, j), fun_type, add_tag, gwth_tag) in enumerate(zip(self.edges_index,
                                                    self.edge_funcs,
                                                    self.add_edge_interaction_bools,
                                                    self.growth_interaction_bools)):
            if (i, j) in self.regular_edges_index:
                if add_tag and gwth_tag:
                    efunc_add_growthterm_vect[j].append(fun_type(self.c_vect_s[i], B_s[ei], n_s[ei]))
                    efunc_mult_growthterm_vect[j].append(None)
                    efunc_mult_decayterm_vect[j].append(None)

                elif not add_tag and gwth_tag:
                    efunc_mult_growthterm_vect[j].append(fun_type(self.c_vect_s[i], B_s[ei], n_s[ei]))
                    efunc_add_growthterm_vect[j].append(None)
                    efunc_mult_decayterm_vect[j].append(None)

                elif not add_tag and not gwth_tag:
                    efunc_mult_decayterm_vect[j].append(fun_type(self.c_vect_s[i], B_s[ei], n_s[ei]))
                    efunc_add_growthterm_vect[j].append(None)
                    efunc_mult_growthterm_vect[j].append(None)
                else:
                    raise Exception("Currently not supporting any other node interaction types.")

        self.efunc_add_growthterm_vect = efunc_add_growthterm_vect
        self.efunc_mult_growthterm_vect = efunc_mult_growthterm_vect
        self.efunc_mult_decayterm_vect = efunc_mult_decayterm_vect

        # Initialize the dcdt_vect_s with zeros:
        dcdt_vect_s = [0 for nni in self.nodes_index]  # equations for rate of change of a node
        eq_c_vect_s = [0 for nni in self.nodes_index]  # null equations for direct solutions for a node

        # Process additive interactions acting on the growth term:
        for nde_i, (fval_add, fval_mult, fval_multd) in enumerate(zip(efunc_add_growthterm_vect,
                                                                   efunc_mult_growthterm_vect,
                                                                   efunc_mult_decayterm_vect)):
            if (np.all(np.asarray(fval_add) == None) and len(fval_add) != 0):
                fsum = 1

            elif len(fval_add) == 0:
                fsum = 1

            else:
                fsum = 0
                for fi in fval_add:
                    if fi is not None:
                        fsum += fi

            # replace the segment in the efunc vect with the sum:
            efunc_add_growthterm_vect[nde_i] = fsum

        # # Process multiplicative interactions acting on the growth term:
            fprodg = 1
            for fi in fval_mult:
                if fi is not None:
                    fprodg = fprodg*fi

            efunc_mult_growthterm_vect[nde_i] = fprodg

        # Process multiplicative interactions acting on the decay term:
            fprodd = 1
            for fi in fval_multd:
                if fi is not None:
                    fprodd = fprodd*fi

            efunc_mult_decayterm_vect[nde_i] = fprodd

            # for ni in range(self.N_nodes): # Creating the sum terms above, construct the equation
            # if we're not dealing with a 'special' node that follows non-Hill dynamics:
            if (nde_i in self.node_type_inds[NodeType.gene.name] or
                    nde_i in self.node_type_inds[NodeType.effector.name]):
                dcdt_vect_s[nde_i] = (d_max_s[nde_i]*efunc_mult_growthterm_vect[nde_i]*efunc_add_growthterm_vect[nde_i]
                                   - self.c_vect_s[nde_i] * d_max_s[nde_i] * efunc_mult_decayterm_vect[nde_i])
            elif nde_i in self.process_node_inds:
                proc_i = self.process_node_inds.index(nde_i) # get the index of the process
                proc_dict = self.processes_list[proc_i] # get the correct info dictionary for the process
                i_eff = proc_dict[NodeType.effector.name] # allow for multiple effectors
                i_fact = proc_dict[NodeType.factor.name][0]
                i_proc = proc_dict[NodeType.process.name]
                i_sens = proc_dict[NodeType.sensor.name][0]

                c_proc = self.c_vect_s[i_proc]
                c_fact = self.c_vect_s[i_fact]

                if len(i_eff) == 1:
                    c_eff = self.c_vect_s[i_eff[0]]
                    # Add the osmotic process attributes to the network:
                    self.osmotic_process(c_eff, c_proc, c_fact)

                elif len(i_eff) == 2:
                    c_eff0 = self.c_vect_s[i_eff[0]]
                    c_eff1 = self.c_vect_s[i_eff[1]]
                    self.osmotic_process(c_eff0, c_proc, c_fact, c_channel=c_eff1)

                else:
                    raise Exception("To define a process must have either 1 or 2 effectors!")

                # and add the process rate equation to the change vector:
                dcdt_vect_s[nde_i] = self.dEdt_s

                # Finally, we need an 'auxillary' equation that represents the value of the
                # sensor in terms of the process, but equals zero (so we can include it in
                # optimization root finding):
                eq_c_vect_s[i_sens] = -self.c_vect_s[i_sens] + f_logi_s(c_proc,
                                                                       self.sensor_params_s[i_sens][0],
                                                                       self.sensor_params_s[i_sens][1])

        # construct an 'extra params' list that holds non-GRN parameter arguments
        # from sensors and the osmotic process:
        self.extra_params_s = []

        for k, v in self.sensor_params_s.items():
            self.extra_params_s.extend(v)

        self.extra_params_s.extend(self.process_params_s)

        self.eq_c_vect_s = sp.Matrix(eq_c_vect_s)

        # analytical rate of change of concentration vector for the network:
        self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)

        self._include_process = False  # Set the internal boolean to True for consistency

        # vector to find roots of in any search (want to find the zeros of this function):
        self.dcdt_vect_s = self.eq_c_vect_s + self.dcdt_vect_s

        # Generate the optimization "energy" function as well as jacobians and hessians for the system:
        self._generate_optimization_functions()
    def _generate_optimization_functions(self):
        '''
        Using the model equations, generate numerical optimization functions
        and rate functions required to solve system properties numerically.
        '''

        if self._reduced_dims and self._solved_analytically is False:
            dcdt_vect_s = self.dcdt_vect_reduced_s
            c_vect_s = self.c_vect_reduced_s

        else:
            dcdt_vect_s = self.dcdt_vect_s
            c_vect_s = self.c_vect_s

        lambda_params = self._fetch_lambda_params_s(self.c_vect_s)
        lambda_params_r = self._fetch_lambda_params_s(c_vect_s)

        # Create a Jacobian for the whole system
        self.jac_s = self.dcdt_vect_s.jacobian(sp.Matrix(self.c_vect_s))

        # The Hessian is a more complex tensor for the whole system:
        self.hess_s = sp.Array(
            [[[self.dcdt_vect_s[i].diff(dcj).diff(dci) for dcj in self.c_vect_s]
              for dci in self.c_vect_s] for i in range(self.N_nodes)])

        # Optimization function for solving the problem (defined in terms of reduced dimensions):
        self.opti_s = (dcdt_vect_s.T*dcdt_vect_s)[0]
        self.opti_jac_s = sp.Array([self.opti_s.diff(ci) for ci in c_vect_s])
        self.opti_hess_s = sp.Matrix(self.opti_jac_s).jacobian(c_vect_s)

        # Lambdify the two outputs so they can be used to study the network numerically:
        # On the whole system:
        flatten_f = np.asarray([fs for fs in self.dcdt_vect_s])
        self.dcdt_vect_f = sp.lambdify(lambda_params, flatten_f)
        self.jac_f = sp.lambdify(lambda_params, self.jac_s)
        self.hess_f = sp.lambdify(lambda_params, self.hess_s)

        # These will automatically become the correct dimensions due to definition of lambda_params_r:
        self.opti_f = sp.lambdify(lambda_params_r, self.opti_s)
        self.opti_jac_f = sp.lambdify(lambda_params_r, self.opti_jac_s)
        self.opti_hess_f = sp.lambdify(lambda_params_r, self.opti_hess_s)

        # For case of reduced dims, we need two additional attributes lambdified:
        # If dims are reduced we also need to lambdify the remaining concentration sets
        if self._reduced_dims and self._solved_analytically is False:
            # This is now the same thing as opti-f:
            flatten_fr = np.asarray([fs for fs in self.dcdt_vect_reduced_s])
            self.dcdt_vect_reduced_f = sp.lambdify(lambda_params_r, flatten_fr)

            # Create a reduced Jacobian:
            self.jac_reduced_s = self.dcdt_vect_reduced_s.jacobian(sp.Matrix(self.c_vect_reduced_s))
            self.jac_reduced_f = sp.lambdify(lambda_params_r, self.jac_reduced_s)

            self.sol_cset_f = {}
            for ci, eqci in self.sol_cset_s.items():
                c_ind = self.c_vect_s.index(ci)
                self.sol_cset_f[c_ind] = sp.lambdify(lambda_params_r, eqci)

    def reduce_model_dimensions(self):
        '''
        Using analytical methods, attempt to reduce the multidimensional
        network equations to as few equations as possible.

        '''

        # Solve the nonlinear system as best as possible:
        nosol = False

        try:
            sol_csetoo = sp.nonlinsolve(self.dcdt_vect_s, self.c_vect_s)
            # Clean up the sympy container for the solutions:
            sol_cseto = list(list(sol_csetoo)[0])

            if len(sol_cseto):

                c_master_i = []  # the indices of concentrations involved in the master equations (the reduced dims)
                sol_cset = {}  # A dictionary of auxillary solutions (plug and play)
                for i, c_eq in enumerate(sol_cseto):
                    if c_eq in self.c_vect_s:  # If it's a non-solution for the term, append it as a non-reduced conc.
                        c_master_i.append(self.c_vect_s.index(c_eq))
                    else:  # Otherwise append the plug-and-play solution set:
                        sol_cset[self.c_vect_s[i]] = c_eq

                master_eq_list = []  # master equations to be numerically optimized (reduced dimension network equations)
                c_vect_reduced = []  # concentrations involved in the master equations

                if len(c_master_i):
                    for ii in c_master_i:
                        # substitute in the expressions in terms of master concentrations to form the master equations:
                        ci_solve_eq = self.dcdt_vect_s[ii].subs([(k, v) for k, v in sol_cset.items()])
                        master_eq_list.append(ci_solve_eq)
                        c_vect_reduced.append(self.c_vect_s[ii])

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
            self.dcdt_vect_reduced_s = None
            self.c_vect_reduced_s = None
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
                self.dcdt_vect_reduced_s = sp.Matrix(master_eq_list)
                # This is the concentration vector that contains the reduced equation concentration variables:
                self.c_vect_reduced_s = c_vect_reduced
                self.c_master_i = c_master_i # indices of concentrations that are being numerically optimized
                self.c_remainder_i = np.setdiff1d(self.nodes_index, self.c_master_i) # remaining conc indices
                self.c_vect_remainder_s = np.asarray(self.c_vect_s)[self.c_remainder_i].tolist() # remaining concs

                # This is the dictionary of remaining concentrations that are in terms of the reduced concentrations,
                # such that when the master equation set is solved, the results are plugged into the equations in this
                # dictionary to obtain solutions for the whole network:
                self.sol_cset_s = sol_cset

                # Create a set of signal node indices to the reduced c_vect array:
                self.signal_reduced_inds = []
                for si in self.signal_node_inds:
                    if si in self.c_master_i:
                        self.signal_reduced_inds.append(self.c_master_i.index(si))

                self.nonsignal_reduced_inds = np.setdiff1d(np.arange(len(self.c_master_i)),
                                                           self.signal_reduced_inds)

            else:
                # Set most reduced system attributes to None:
                self.dcdt_vect_reduced_s = None
                self.c_vect_reduced_s = None
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

            # Generate the optimization "energy" function as well as jacobians and hessians for the system.
            # self.sol_cset_s is lambdified in the following method:
            self._generate_optimization_functions()

    def create_parameter_vects(self, Bi: float|list=2.0,
                                ni:float|list=3.0,
                                di:float|list=1.0,
                                co: float|list = 0.0,
                                ki: float|list = 10.0
                                ):
        '''
        Create parameter vectors for the rate equations of the model nodes.
        If floats are specified, use the same parameters for all edges and nodes in the network; if lists
        are specified use them to create the model parameter vectors.

        '''
        # :
        if type(Bi) is not list:
            B_vect = []
            for ei in range(self.N_edges):
                B_vect.append(Bi)

        else:
            B_vect = Bi

        if type(ni) is not list:
            n_vect = []
            for ei in range(self.N_edges):
                n_vect.append(ni)
        else:
            n_vect = ni

        if type(di) is not list:
            d_vect = []
            for nde_i in range(self.N_nodes):
                d_vect.append(di)
        else:
            d_vect = di

        if type(ki) is not list:
            k_vect = []
            for nde_i in self.sensor_node_inds:
                k_vect.append(ki)

        else:
            k_vect = ki

        if type(co) is not list:
            co_vect = []
            for nde_i in self.sensor_node_inds:
                co_vect.append(co)

        else:
            co_vect = co

        self.B_vect = B_vect
        self.n_vect = n_vect
        self.d_vect = d_vect

        self.extra_params_f = []
        for cc, kk in zip(co_vect, k_vect):
            self.extra_params_f.extend((cc, kk))

        if len(self.process_params_s):
            self.extra_params_f.extend(self.process_params_f)

    def _fetch_lambda_params_s(self, cvect_s: list, cko_s: Symbol|Indexed|None = None) -> tuple:
        '''
        Obtain the correct set of parameters to use
        when creating numerical lambda functions from
        sympy analytical functions.
        '''
        if cko_s is None:
            lambda_params = [cvect_s,
                               self.d_vect_s,
                               self.B_vect_s,
                               self.n_vect_s]
        else:
            lambda_params = [cvect_s,
                             cko_s,
                               self.d_vect_s,
                               self.B_vect_s,
                               self.n_vect_s]

        if len(self.process_node_inds) or len(self.sensor_node_inds):
            lambda_params.append(self.extra_params_s)

        return tuple(lambda_params)

    def _fetch_function_args_f(self, cko: float | None = None) -> tuple:
        '''
        Obtain the correct set of auxiliary parameters (arguments) to use
        when evaluating numerical lambda functions created
        from sympy analytical functions in optimization functions.
        '''
        if cko is None:
            lambda_params = [self.d_vect,
                             self.B_vect,
                             self.n_vect]
        else:
            lambda_params = [cko,
                             self.d_vect,
                             self.B_vect,
                             self.n_vect]

        if len(self.process_node_inds) or len(self.sensor_node_inds):
            lambda_params.append(self.extra_params_f)

        return tuple(lambda_params)

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
                self.node_types.append(NodeType.root)
            elif nde_t == 'Path':
                self.node_types.append(NodeType.path)
            else:
                raise Exception("Node type not found.")

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

    def generate_state_space(self,
                             c_vect_s: list|ndarray,
                             Ns: int=3,
                             cmin: float=0.0,
                             cmax: float=1.0,
                             include_signals: bool = False  # include signal node states in the search?
                             ):
        '''
        
        '''
        c_test_lin_set = []

        if self._reduced_dims and self._solved_analytically is False:
            signal_inds = self.signal_reduced_inds
            nonsignal_inds = self.nonsignal_reduced_inds
        else:
            signal_inds = self.signal_node_inds
            nonsignal_inds = self.nonsignal_node_inds

        if include_signals is False:
            # Create a c_vect sampled to the non-signal nodes:
            c_vect = np.asarray(c_vect_s)[nonsignal_inds].tolist()

        else:
            c_vect = c_vect_s # otherwise use the whole vector

        for nd_i, ci in enumerate(c_vect):
            i = c_vect.index(ci)
            if i in self.process_node_inds:
                c_test_lin_set.append(np.linspace(self.strain_min, self.strain_max, Ns))
            else:
                c_test_lin_set.append(np.linspace(cmin, cmax, Ns))

        # Create a set of matrices specifying the concentration grid for each
        # node of the network:
        C_test_M_SET = np.meshgrid(*c_test_lin_set, indexing='ij')

        # Create linearized arrays for each concentration, stacked into one column per node:
        c_test_set = np.asarray([cM.ravel() for cM in C_test_M_SET]).T

        if include_signals is False:  # then we need to add on a zeros block for signal state
            n_rows_test = c_test_set.shape[0]
            signal_block = np.zeros((n_rows_test, len(signal_inds)))
            c_test_set = np.column_stack((c_test_set, signal_block))

        return c_test_set, C_test_M_SET, c_test_lin_set


    def optimized_phase_space_search(self,
                                     Ns: int=3,
                                     cmin: float=0.0,
                                     cmax: float=1.0,
                                     c_bounds: list|None = None,
                                     tol:float = 1.0e-15,
                                     round_sol: int=6, # decimals to round solutions to prevent duplicates
                                     method: str='Root', # Solve by finding the roots of the dc/dt equation
                                     include_signals: bool = False  # include signal node states in the search?
                                     ):
        '''

        '''

        if self.dcdt_vect_f is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        # Determine the set of additional arguments to the optimization function:
        function_args = self._fetch_function_args_f()

        # Initialize the equillibrium point solutions to be a set:
        mins_found = []

        # If it's already been solved analytically, we can simply plug in the variables to obtain the solution
        # at the minimum rate:
        if self._solved_analytically:
            mins_foundo = [[] for i in range(self.N_nodes)]
            for ii, eqi in self.sol_cset_f.items():
                mins_foundo[ii] = eqi(*function_args)

            mins_found.append(mins_foundo)

        else: # if we don't have an explicit solution:
            # Otherwise, we need to go through the whole optimization:
            if self._reduced_dims:
                N_nodes = len(self.c_vect_reduced_s)
                c_vect_s = self.c_vect_reduced_s
                dcdt_funk = self.dcdt_vect_reduced_f
            else:
                N_nodes = self.N_nodes
                c_vect_s = self.c_vect_s
                dcdt_funk = self.dcdt_vect_f

            # Generate the points in state space to sample at:
            c_test_set, _, _ = self.generate_state_space(c_vect_s,
                                                         Ns=Ns,
                                                         cmin=cmin,
                                                         cmax=cmax,
                                                         include_signals=include_signals)

            if c_bounds is None:
                c_bounds = [(cmin, cmax) for i in range(N_nodes)]

                if len(self.process_node_inds):
                    if self._reduced_dims is False:
                        for ip in self.process_node_inds:
                            c_bounds[ip] = (self.strain_min, self.strain_max)
                else:
                    for ip in self.process_node_inds:
                        if self.c_vect_s[ip] in self.c_vect_reduced_s:
                            i = self.c_vect_reduced_s.index(self.c_vect_s[ip])
                            c_bounds[i] = (self.strain_min, self.strain_max)

            self._c_bounds = c_bounds # save for inspection later

            for c_vecti in c_test_set:
                if method == 'Powell' or method == 'trust-constr':
                    if method == 'Powell':
                        jac = None
                        hess = None
                    else:
                        jac = self.opti_jac_f
                        hess = self.opti_hess_f

                    sol0 = minimize(self.opti_f,
                                    c_vecti,
                                    args=function_args,
                                    method=method,
                                    jac=jac,
                                    hess=hess,
                                    bounds=c_bounds,
                                    tol=tol,
                                    callback=None,
                                    options=None)

                    mins_found.append(sol0.x)

                else:
                    sol_root = fsolve(dcdt_funk, c_vecti, args=function_args, xtol=tol)
                    if len(self.process_node_inds) == 0:
                        # If we're not using the process, constrain all concs to be above zero
                        if (np.all(np.asarray(sol_root) >= 0.0)):
                            mins_found.append(sol_root)
                    else:
                        # get the nodes that must be constrained above zero:
                        conc_nodes = np.setdiff1d(self.nodes_index, self.process_node_inds)
                        # Then, only the nodes that are gene products must be above zero
                        if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
                            mins_found.append(sol_root)

            if self._reduced_dims is False:
                self.mins_found = mins_found

            else: # we need to piece together the full solution as the minimum will only be a subset of all
                # concentrations
                full_mins_found = []
                for mins_foundi in list(mins_found): # for each set of unique minima found
                    mins_foundo = [[] for i in range(self.N_nodes)]
                    for cmi, mi in zip(self.c_master_i, mins_foundi):
                        for ii, eqi in self.sol_cset_f.items():
                            mins_foundo[ii] = eqi(mins_foundi, *function_args) # compute the sol for this conc.
                        # also add-in the minima for the master concentrations to the full list
                        mins_foundo[cmi] = mi
                    # We've redefined the mins list so it now includes the full set of concentrations;
                    # flatten the list and add it to the new set:
                    full_mins_found.append(mins_foundo)

                # Redefine the mins_found set for the full concentrations
                mins_found = full_mins_found

        # ensure the list is unique:
        mins_found = np.round(mins_found, round_sol)
        mins_found = np.unique(mins_found, axis=0).tolist()

        return mins_found

    def stability_estimate(self,
                           mins_found: set|list,
                           ):
        '''

        '''

        eps = 1.0e-25 # we need a small value to add to avoid dividing by zero

        sol_dicts_list = []
        # in some Jacobians
        for cminso in mins_found:

            solution_dict = {}

            # print(f'min vals: {cminso}')
            solution_dict['Minima Values'] = cminso

            cmins = np.asarray(cminso) + eps # add the small amount here, before calculating the jacobian

            func_args = self._fetch_function_args_f()

            # print(f'dcdt at min: {self.dcdt_vect_f(cmins, r_vect, d_vect, K_vect, n_vect)}')

            solution_dict['Change at Minima'] = self.dcdt_vect_f(cmins, *func_args)

            jac = self.jac_f(cmins, *func_args)

           # get the eigenvalues of the jacobian at this equillibrium point:
            eig_valso, eig_vects = np.linalg.eig(jac)

            # round the eigenvalues so we don't have issue with small imaginary components
            eig_vals = np.round(np.real(eig_valso), 1) + np.round(np.imag(eig_valso), 1)*1j

            # print(f'Jacobian eigs: {eig_vals}')

            solution_dict['Jacobian Eigenvalues'] = eig_vals

            # get the indices of eigenvalues that have only real components:
            real_eig_inds = (np.imag(eig_vals) == 0.0).nonzero()[0]
            # print(real_eig_inds)

            # If all eigenvalues are real and they're all negative:
            if len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) <= 0.0):
                # print('Stable Attractor')
                char_tag = 'Stable Attractor'

            # If all eigenvalues are real and they're all positive:
            elif len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) > 0.0):
                # print('Stable Repellor')
                char_tag = 'Stable Repellor'

            # If there are no real eigenvalues we only know its a limit cycle but can't say
            # anything certain about stability:
            elif len(real_eig_inds) == 0 and np.all(np.real(eig_vals) <= 0.0):
                # print('Stable Limit cycle')
                char_tag = 'Stable Limit Cycle'

            # If there are no real eigenvalues and a mix of real component sign, we only know its a limit cycle but can't say
            # anything certain about stability:
            elif len(real_eig_inds) == 0 and np.any(np.real(eig_vals) > 0.0):
                # print('Limit cycle')
                char_tag = 'Limit Cycle'

            elif np.all(np.real(eig_vals[real_eig_inds]) <= 0.0):
                # print('Stable Limit Cycle')
                char_tag = 'Stable Limit Cycle'

            elif np.any(np.real(eig_vals[real_eig_inds] > 0.0)):
                # print('Saddle Point')
                char_tag = 'Saddle Point'
            else:
                # print('Undetermined Stability Status')
                char_tag = 'Undetermined'

            solution_dict['Stability Characteristic'] = char_tag

            sol_dicts_list.append(solution_dict)

            # print('----')
        self.sol_dicts_list = sol_dicts_list

        return sol_dicts_list


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

    def pulses(self,
               tvect: list|ndarray,
               t_on: float|int,
               t_off: float|int,
               p_max: float|int = 1.0,
               p_min: float|int = 0.0,
               f_pulses: float|int|None = None,
               duty_cycle: float = 0.1):
        '''

        '''
        itop = (tvect >= t_on).nonzero()[0]
        ibot = (tvect <= t_off).nonzero()[0]

        ipulse = np.intersect1d(ibot, itop)

        pulse_sig = np.zeros(len(tvect))

        if f_pulses is None:
            pulse_sig[ipulse] = p_max
            pulse_sig += p_min

        else:
            pulse_sig[ipulse] = p_max*square(2 * np.pi * tvect[ipulse] * f_pulses, duty=duty_cycle)
            pulse_sig[(pulse_sig < 0.0).nonzero()[0]] = 0.0

            pulse_sig += p_min

        return pulse_sig

    def make_signals_matrix(self,
                            tvect: list|ndarray,
                            sig_inds: list|ndarray,
                            sig_times: list|ndarray,
                            sig_mag: list|ndarray):
        '''

        '''
        Nt = len(tvect)

        c_signals = np.zeros((Nt, self.N_nodes))  # Initialize matrix holding the signal sequences

        for si, (ts, te), (smin, smax) in zip(sig_inds, sig_times, sig_mag):
            c_signals[:, si] += self.pulses(tvect, ts, te, p_max=smax, p_min=smin, f_pulses=None, duty_cycle=0.1)

        return c_signals

    def run_time_sim(self,
                     tend: float,
                     dt: float,
                     cvecti: ndarray|list,
                     sig_inds: ndarray|list|None = None,
                     sig_times: ndarray | list | None = None,
                     sig_mag: ndarray | list | None = None,
                     dt_samp: float|None = None
                     ):
        '''

        '''
        Nt = int(tend/dt)
        tvect = np.linspace(0.0, tend, Nt)

        if sig_inds is not None:
            c_signals = self.make_signals_matrix(tvect, sig_inds, sig_times, sig_mag)
        else:
            c_signals = None

        concs_time = []

        # sampling compression
        if dt_samp is not None:
            sampr = int(dt_samp / dt)
            tvect_samp = tvect[0::sampr]
            tvectr = tvect_samp
        else:
            tvect_samp = None
            tvectr = tvect

        # make a time-step update vector so we can update any sensors as
        # an absolute reading (dt = 1.0) while treating the kinetics of the
        # other node types:
        dtv = 1.0e-3 * np.ones(self.N_nodes)
        dtv[self.sensor_node_inds] = 1.0

        function_args = self._fetch_function_args_f()

        for ti, tt in enumerate(tvect):
            dcdt = self.dcdt_vect_f(cvecti, *function_args)
            cvecti += dtv * dcdt

            if c_signals is not None:
                # manually set the signal node values:
                cvecti[self.signal_node_inds] = c_signals[ti, self.signal_node_inds]

            if dt_samp is None:
                concs_time.append(cvecti * 1)
            else:
                if tt in tvect_samp:
                    concs_time.append(cvecti * 1)

        concs_time = np.asarray(concs_time)

        return concs_time, tvectr

    def find_attractor_sols(self,
                            sols_0: list|ndarray,
                            tol: float=1.0e-3,
                            verbose: bool=True,
                            N_round: int = 12,
                            unique_sols: bool = False,
                            sol_round: int = 1,
                            save_file: str|None = None
                            ):
        '''

        '''

        sol_char_0 = self.stability_estimate(sols_0)

        solsM = []
        sol_char_list = []
        sol_char_error = []
        i = 0
        for sol_dic in sol_char_0:
            error = np.sum(sol_dic['Change at Minima'])**2
            char = sol_dic['Stability Characteristic']
            sols = sol_dic['Minima Values']

            if char != 'Saddle Point' and error <= tol:
                i += 1
                if verbose and unique_sols is False:
                    print(f'Soln {i}, {char}, {sols}, {np.round(error, N_round)}')
                solsM.append(sols)
                sol_char_list.append(char)
                sol_char_error.append(error)

        solsM_return = np.asarray(solsM).T

        if unique_sols and len(solsM) != 0:
            # round the sols to avoid degenerates and return indices to the unique solutions:
            solsy, inds_solsy = np.unique(np.round(solsM, sol_round), axis=0, return_index=True)
            if verbose:
                for i, si in enumerate(inds_solsy):
                    print(f'Soln {i}: {sol_char_list[si]}, {solsM[si]}, error: {sol_char_error[si]}')

            solsM_return = np.asarray(solsM)[inds_solsy].T

        if save_file is not None:
            solsMi = np.asarray(solsM)
            header = [f'State {i}' for i in range(solsMi.shape[0])]
            with open(save_file, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # create a csvwriter object
                csvwriter.writerow(header)  # write the header
                csvwriter.writerow(sol_char_error)  # write the root error at steady-state
                csvwriter.writerow(sol_char_list)  # write the attractor characterization
                for si in solsMi.T:
                    csvwriter.writerow(si)  # write the soln data rows for each gene

        return solsM_return

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
                             save_eqn_csv: str|None = None
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
        t_s = sp.symbols('t')
        c_change = sp.Matrix([sp.Derivative(ci, t_s) for ci in self.c_vect_s])
        eqn_net = sp.Eq(c_change, self.dcdt_vect_s)

        sp.preview(eqn_net,
                   viewer='file',
                   filename=save_eqn_image,
                   euler=False,
                   dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600", "-bg", "Transparent"])

        # Save the equations for the graph to a file:
        header = ['Concentrations', 'Change Vector']
        eqns_to_write = [[sp.latex(self.c_vect_s), sp.latex(self.dcdt_vect_s)]]

        if self.dcdt_vect_reduced_s is not None and save_reduced_eqn_image is not None:
            c_change_reduced = sp.Matrix([sp.Derivative(ci, t_s) for ci in self.c_vect_reduced_s])
            eqn_net_reduced = sp.Eq(c_change_reduced, self.dcdt_vect_reduced_s)

            sp.preview(eqn_net_reduced,
                       viewer='file',
                       filename=save_reduced_eqn_image,
                       euler=False,
                       dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 600", "-bg", "Transparent"])

            eqns_to_write.append(sp.latex(self.c_vect_reduced_s))
            eqns_to_write.append(sp.latex(self.dcdt_vect_reduced_s))
            header.extend(['Reduced Concentations', 'Reduced Change Vector'])

        if save_eqn_csv is not None:
            with open(save_eqn_csv, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # 2. create a csvwriter object
                csvwriter.writerow(header)  # 4. write the header
                csvwriter.writerows(eqns_to_write)  # 5. write the rest of the data

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





