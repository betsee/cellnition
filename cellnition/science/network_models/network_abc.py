#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module is a base class for continuous and Boolean regulatory network models.
'''
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import networkx as nx
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             )
import pygraphviz as pgv

class NetworkABC(object):
    '''
    This baseclass allows one to generate a regulatory network as a graph using a
    procedural construction algorithm, or from user-input edges. It can then perform
    analysis on the resulting graph to determine cycles, input and output degree distributions,
    hierarchical attributes, and other characteristics. The baseclass comes equipt with
    various visualization methods to plot network degree distributions and visualize state heatmaps.

    Attributes
    ----------
    N_edges : int
        Total number of edges in the regulatory network.
    N_nodes : int
        Total number of nodes in the regulatory network.
    edges_list : list[tuples]
        List of the edges as tuples containing node names.
    nodes_list : list
        List of the nodes by numerical node index.
    edges_index : list
        List of the edges as edge indices.
    nodes_index : list
        List of the nodes by node name.
    GG : networkx.DiGraph
        The regulatory network graph as a networkx.DiGraph object.
    selfloop_edge_inds : list
        The edge indices that are self-loops (i.e. node A --> node A).
    in_degree_sequence : list
        The in-degree sequence of nodes, arranged according to node index order.
    out_degree_sequence : list
        The out-degree sequence of nodes, arranged according to node index order.
    in_dmax : int
        The maximum in-degree of the regulatory network.
    out_dmax : int
        The maximum out-degree of the regulatory network.
    node_divergence : list
        The divergence of each node, as the difference between in- and out- degree.
    in_bins : list
        Bins used to count of how many nodes have each binned in-degree (used to plot degree distribution histograms).
    in_degree_counts : list
        A count of how many nodes have each in-degree bin (used to plot degree distribution histograms).
    out_bins : list
        Bins used to count of how many nodes have each binned out-degree (used to plot degree distribution histograms).
    out_degree_counts : list
        A count of how many nodes have each out-degree bin (used to plot degree distribution histograms).
    nodes_by_out_degree : list
        Nodes arranged according to the number of outputs (out degree).
    nodes_by_in_degree : list
        Nodes arranged according to the number of inputs (in degree).
    graph_cycles : list(tuple)
        Cycles of the regulatory network, as defined by network nodes.
    N_cycles : int
        Number of simple cycles detected in the regulatory network.
    nodes_in_cycles : list
        Nodes of the regulatory network that do participate in cycles.
    nodes_acyclic : list
        Nodes of the regulatory network that do not participate in cycles.
    hier_node_level : list
        Overall hierarchical node levels of the graph (this is akin to a y-coordinate for each node of the network).
    dem_coeff : float
        The democracy coefficient parameter, measuring how much the influencers of a graph are influenced
        themselves.
    hier_incoherence : float
        The hierarchical incoherence parameter, measuring how much feedback there is in the network, with higher
        levels indicating more feedback, and lower levels indicating more hierarchy.
    input_node_inds : list[int]
        These are nodes with zero in degree, which represent the input nodes of the regulatory network.
    output_node_inds : list[int]
        These are nodes with zero out degree, which represent output nodes and potential effectors.
    main_nodes : list[int]
        The main nodes of the regulatory network, which are nodes that are neither input nor output nodes
        (these are the internal nodes).
    sensor_node_inds : list[int]
        User-defined nodes with NodeType.sensor node types.
    process_node_inds : list[int]
        User-defined nodes with NodeType.process node types.
    noninput_node_inds : list[int]
        Nodes of the network excluding the input nodes, but still representing internal and output nodes.
    factor_node_inds : list[int]
        User-defined nodes with NodeType.factor node types.

    '''

    def __init__(self) -> None:
        '''
        Initialize the class to begin building and characterizing a regulatory network graph.
        '''
        pass


    def build_network_from_edges(self, edges: list[tuple]):
        '''
        Use a list of tuples defining directed edges between nodes to
        build a regulatory network.

        Parameters
        ----------
        edges : list[tuple]
            List with tuples defining each directed edge in the regulatory
            network as passing from the first to second node in the tuple.

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
        Procedurally generate a network with a scale-free or binomial (random) degree distribution.

        Parameters
        ----------
        N_nodes : int
            The number of nodes to build the network (only used in randomly built networks, otherwise the number of
            nodes is calculated from the number of unique nodes supplied in the edges list).
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
        this method creates numerical (i.e. integer) indices for the nodes and
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

        Parameters
        ----------
        count_cycles : bool, default: True
            Do you wish to perform a cycle count of the network (True)? Some regulatory networks have
            very high numbers of cycles and in this case the cycle count can consume all the memory and
            should therefore be disabled.
        cycle_length_bound : int|None, default: None
            Specify an upper bound for the length of a cycle in a network in terms of node number (e.g. 12).
            For networks with large cycle numbers, specifying an upper bound can prevent the extreme counts
            that would otherwise be produced.

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

    def get_edge_types(self, p_acti: float=0.5, set_selfloops_acti: bool=True) -> list:
        '''
        Automatically generate a list of EdgeType for use in model building.
        The edge type specifies whether the edge is an activating or inhibiting
        relationship between the nodes. This routine randomly chooses a set of
        activating and inhibiting edge types for a model.

        Parameters
        ----------
        p_acti : float, default: 0.5
            The probability of an edge type being an activator. Note that this value
            must be less than 1.0, and that the probability of an edge being an
            inhibitor becomes 1.0 - p_acti.

        set_selfloops_acti : bool, default: True
            Work shows that, in general, self-inhibition does not generate models with
            multistable states. Therefore, this edge-type assignment routine allows one to
            automatically set all self-loops to be activation interactions.

        Returns
        -------
        list
            A list containing an EdgeType enum for every edge in the network.

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
        Assign a list EdgeType to edges of the graph.

        Parameters
        ----------
        edge_types : list|ndarray
            A list of edge type enumerations; one for each edge of the network.
        '''
        self.edge_types = edge_types

        # assign the edge types to the graph in case we decide to save the network:
        edge_attr_dict = {}
        for ei, et in zip(self.edges_list, edge_types):
            edge_attr_dict[ei] = {"edge_type": et.name}

        nx.set_edge_attributes(self.GG, edge_attr_dict)

    def set_node_types(self, node_type_dict: dict|None = None, pure_gene_edges_only: bool = False):
        '''
        Assign a dictionary of NodeType to nodes of the graph.

        Parameters
        ----------
        node_type_dict : dict|None, default: None
            A list of node type enumerations for each node of the network.
        pure_gene_edges_only : bool, default: False
            Classify multiple non-process NodeType as "genes" (True) or only NodeType.gene (False).
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

    def save_network(self, filename: str):
        '''
        Write a network, including edge types, to a saved file.

        Parameters
        ----------
        filename : str
            The full directory and filename to write the graph file to
            as a gml format graph.
        '''
        nx.write_gml(self.GG, filename)

    def save_network_image(self, save_filename: str, use_dot_layout: bool=False):
        '''
        Uses pygraphviz to create a basic plot of the network model.

        Parameters
        ----------
        save_filename : str
            The full directory and filename to write the graph image file to.
            If the filename ends with '.png' the image will be a raster image, if it ends
            with '.svg' it will be a vector graphics file.
        use_dot_layout : bool, default: false
            Use the 'dot' layout to build the graph.

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

    def plot_degree_distributions(self) -> tuple[object, object]:
        '''
        Generate a plot of the in- and out- degree distributions of the
        network as histograms. Requires self.characterize_graph() to have
        been run previously.

        Returns
        -------
        fig : matplotlib.figure
        ax : matplotlib.axes

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
                        figsize: tuple=(10,10)) -> tuple[object, object]:
        '''
        Create and save a heatmap image representing a matrix of states for the
        regulatory network.

        Parameters
        ----------
        solsM : ndarray
            The matrix of regulatory network states, with each state being a coloumn
            of the matrix and each row representing the expression level of a node in
            the network.
        gene_inds : list|ndarray|None, default: None
            A subset of the total nodes of the network that are to be displayed in the
            visualized heatmap.
        figsave : str|None, default: None
            The full directory and filename to write the image file to. If None, no image
            will be save to disk.
        cmap : str|None, default: None
            The matplotlib colormap to use for the image.
        save_format : str, default: 'png'
            The file format to save the image in ('svg' or 'png').
        figsize : tuple, default: (10,10)
            The size of the figure.

        Returns
        -------
        fig : matplotlib.figure
        ax : matplotlib.axes

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
                          fontsize: int=16) -> tuple[object, object]:
        '''
        Plot a matrix of values as a heatmap.

        Parameters
        ----------
        solsM : ndarray
            The matrix of values to plot as a heatmap.
        x_labels : list|ndarray|None
            Labels to apply to each column of the solsM matrix, along the horizontal axis.
        y_labels : list|ndarray|None
            Labels to apply to each row of the solsM matrix, along the vertical axis.
        figsave : str|None, default: None
            The full directory and filename to write the image file to. If None, no image
            will be save to disk. Only 'png' images can be exported.
        cmap : str|None, default: None
            The matplotlib colormap to use for the image.
        cbar_label : str, default: ''
            The text label to write along the image colorbar.
        figsize : tuple, default: (10,10)
            The size of the figure.

        Returns
        -------
        fig : matplotlib.figure
        ax : matplotlib.axes

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



