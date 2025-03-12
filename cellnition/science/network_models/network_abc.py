#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module is an abstract base class for network models.
'''
from abc import ABCMeta
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             InterFuncType)

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

    def __init__(self
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
                                          N_nodes: int,
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
        self.N_nodes = N_nodes
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

        self._make_node_edge_indices()

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

    def characterize_graph(self, count_cycles: bool=True, cycle_length_bound: int|None=None):
        '''
        Perform a number of graph-theory style characterizations on the network to determine
        cycle number, analyze in- and out- degree distribution, and analyze hierarchy. Hierarchical
        structure analysis was from the work of Moutsinas, G. et al. Scientific Reports 11 (2021).

        '''

        # print('Degree sequences...')
        # Indices of edges with selfloops:
        self.selfloop_edge_inds = [self.edges_list.index(ei)
                                   for ei in list(nx.selfloop_edges(self.GG))]

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

        if count_cycles:
            # print('Cycle Number...')
            # Number of cycles:
            self.graph_cycles = sorted(nx.simple_cycles(self.GG, length_bound=cycle_length_bound))
            self.N_cycles = len(self.graph_cycles)

            # Determine the nodes in the cycles:
            nodes_in_cycles = set()
            for nde_lst in self.graph_cycles:
                for nde_ni in nde_lst:
                    nde_i = self.nodes_list.index(nde_ni)
                    nodes_in_cycles.add(nde_i)

            self.nodes_in_cycles = list(nodes_in_cycles)
            self.nodes_acyclic = np.setdiff1d(self.nodes_index, nodes_in_cycles)

        # print('Graph hierarchical structure...')
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
            # rev hierarchical levels grade vertices based on distance from sink subgraphs
            self.rev_hier_node_level = L_out_inv.dot(self.out_degree_sequence)
            # overall hierarchical levels of the graph (this is akin to a y-coordinate for each node of the network):
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

            # I don't think this is calculated correctly...
            self.influence_centrality = (1 -
                                         self.D_diff.T.dot(self.fwd_hier_diff)/(1 +
                                                          np.asarray(self.in_degree_sequence))
                                         )

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

        self.output_node_inds = ((np.asarray(self.out_degree_sequence) == 0).nonzero()[0]).tolist()
        self.sensor_node_inds = self.node_type_inds[NodeType.sensor.name]
        self.process_node_inds = self.node_type_inds[NodeType.process.name]

        if len(self.node_type_inds[NodeType.effector.name]) == 0:
            self.effector_node_inds = ((np.asarray(self.out_degree_sequence) == 0).nonzero()[0]).tolist()
        else:
            self.effector_node_inds = self.node_type_inds[NodeType.effector.name]

        self.noninput_node_inds = np.setdiff1d(self.nodes_index, self.input_node_inds).tolist()

        self.factor_node_inds = self.node_type_inds[NodeType.factor.name]

        # also determine the "main nodes", which are nodes that are not input and also are not effectors:
        self.main_nodes = np.setdiff1d(self.noninput_node_inds, self.effector_node_inds).tolist()

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

    def plot_sols_array(self,
                        solsM: ndarray,
                        gene_inds: list|ndarray|None=None,
                        figsave: str | None = None,
                        cmap: str | None =None,
                        save_format: str='png',
                        figsize: tuple=(10,10)):
        '''

        '''

        if cmap is None:
            cmap = 'magma'

        state_labels = [f'State {i}' for i in range(solsM.shape[1])]

        if gene_inds is None:
            gene_labels = np.asarray(self.nodes_list)

        else:
            gene_labels = np.asarray(self.nodes_list)[gene_inds]

        fig, ax = plt.subplots(figsize=figsize)

        if gene_inds is None:
            im = ax.imshow(solsM, cmap=cmap)
        else:
            im = ax.imshow(solsM[gene_inds, :], cmap=cmap)

        ax.set_xticks(np.arange(len(state_labels)), labels=state_labels)
        ax.set_yticks(np.arange(len(gene_labels)), labels=gene_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label='Expression Level')

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format=save_format)

        return fig, ax

    def plot_pixel_matrix(self,
                          solsM: ndarray,
                          x_labels: list | ndarray|None,
                          y_labels: list|ndarray|None,
                          figsave: str | None = None,
                          cmap: str | None = None,
                          cbar_label: str = '',
                          figsize: tuple = (10, 10),
                          fontsize: int=16):
        '''
        Plot a correlation or adjacency matrix for a subset of genes.

        '''

        if cmap is None:
            cmap = 'magma'

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(solsM, cmap=cmap)
        ax.set_xticks(np.arange(solsM.shape[1]), labels=x_labels, font='DejaVu Serif', fontsize=fontsize)
        ax.set_yticks(np.arange(solsM.shape[0]), labels=y_labels, fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label=cbar_label)

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format='png')

        return fig, ax



