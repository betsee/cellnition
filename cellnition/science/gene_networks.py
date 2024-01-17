#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module
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
from cellnition.science.enumerations import EdgeType, GraphType, NodeType
from cellnition.science.stability import Solution
import pygraphviz as pgv
import pyvista as pv

# TODO: Parameter scaling module: scale K and d by 's' and apply rate scaling 'v'
# TODO: Allow multiple effectors to be added to the network
# TODO: Plot a path through a graph

class GeneNetworkModel(object):
    '''

    '''

    def __init__(self,
                 N_nodes: int,
                 edges: list|ndarray|None = None,
                 graph_type: GraphType = GraphType.scale_free,
                 beta: float = 0.20,
                 gamma: float=0.75,
                 delta_in: float=0.0,
                 delta_out: float = 0.0,
                 p_edge: float=0.5):
        '''

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


        self._reduced_dims = False # Indicate that model is full dimensions
        self._include_process = False # Indicate that model does not include the process by default
        self._solved_analytically = False # Indicate that the model does not have an analytical solution
        self.dcdt_vect_reduced_s = None

    def generate_network(self,
                         beta: float=0.15,
                         gamma: float=0.8,
                         delta_in: float=0.0,
                         delta_out: float=0.0,
                         p_edge: float=0.5,
                         graph_type: GraphType = GraphType.scale_free
                         ):
        '''

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

    def characterize_graph(self):
        '''

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
            for nde_i in nde_lst:
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

    def get_paths_matrix(self):

        # Matrix showing the number of paths from starting node to end node:
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
        Automatically generate a conse
        rved edge-type vector for use in
        model building.
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
        Assign edge_types to the graph and create an edge function list.
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
                self.edge_funcs.append(self.f_acti_s)
                self.add_edge_interaction_bools.append(add_interactions)
                self.growth_interaction_bools.append(True) # interact with growth component of reaction
            elif et is EdgeType.I:
                self.edge_funcs.append(self.f_inhi_s)
                self.add_edge_interaction_bools.append(add_interactions)
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction
            elif et is EdgeType.N:
                self.edge_funcs.append(self.f_neut_s)
                self.add_edge_interaction_bools.append(False)
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction
            elif et is EdgeType.As: # else if it's As for a sensor activation interaction, then:
                # self.edge_funcs.append(self.f_acti_s) # activate
                # self.add_edge_interaction_bools.append(True) # add
                # self.growth_interaction_bools.append(True)  # interact with growth component of reaction
                self.edge_funcs.append(self.f_inhi_s) # inhibit
                self.add_edge_interaction_bools.append(False) # multiply
                self.growth_interaction_bools.append(False)  # interact with decay component of reaction
            else:
                self.edge_funcs.append(self.f_inhi_s) # inhibit
                self.add_edge_interaction_bools.append(False) # multiply
                self.growth_interaction_bools.append(True)  # interact with growth component of reaction

    def set_node_types(self, node_types: list|ndarray):
        '''
        Assign node types to the graph.
        '''
        self.node_types = node_types
        # Set node type as graph node attribute:
        node_attr_dict = {}
        for nde_i, nde_t in zip(self.nodes_index, node_types):
            node_attr_dict[nde_i] = {"node_type": nde_t.value}

        nx.set_node_attributes(self.GG, node_attr_dict)

    def edges_from_path(self, path_nodes: list|ndarray):
        '''

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
                               add_interactions: bool=False):
        '''

        '''

        self._reduced_dims = False # always build models in full dimensions

        if edge_types is None:
            self.edge_types = self.get_edge_types(p_acti=prob_acti)

        else:
            self.edge_types = edge_types

        self.set_edge_types(self.edge_types, add_interactions)

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_index]  # First set all nodes
        # to the gene type

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

        # Determine the node indices of any signal nodes:
        self.signal_inds = []
        for nde_i, nde_t in enumerate(self.node_types):
            if nde_t is NodeType.signal:
                self.signal_inds.append(nde_i)

        self.nonsignal_inds = np.setdiff1d(self.nodes_index, self.signal_inds)

        c_s = sp.IndexedBase('c')
        K_s = sp.IndexedBase('K')
        n_s = sp.IndexedBase('n')
        r_max_s = sp.IndexedBase('r_max')
        d_max_s = sp.IndexedBase('d_max')

        # These are needed for lambdification of analytical models:
        self.K_vect_s = [K_s[i] for i in range(self.N_edges)]
        self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
        self.r_vect_s = [r_max_s[i] for i in self.nodes_index]
        self.d_vect_s = [d_max_s[i] for i in self.nodes_index]
        self.c_vect_s = [c_s[i] for i in self.nodes_index]

        efunc_add_growthterm_vect = [[] for i in self.nodes_index]
        efunc_mult_growthterm_vect = [[] for i in self.nodes_index]
        efunc_mult_decayterm_vect = [[] for i in self.nodes_index]
        for ei, ((i, j), fun_type, add_tag, gwth_tag) in enumerate(zip(self.edges_index,
                                                    self.edge_funcs,
                                                    self.add_edge_interaction_bools,
                                                    self.growth_interaction_bools)):
            if add_tag and gwth_tag:
                efunc_add_growthterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
                efunc_mult_growthterm_vect[j].append(None)
                efunc_mult_decayterm_vect[j].append(None)

            elif not add_tag and gwth_tag:
                efunc_mult_growthterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
                efunc_add_growthterm_vect[j].append(None)
                efunc_mult_decayterm_vect[j].append(None)

            elif not add_tag and not gwth_tag:
                efunc_mult_decayterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
                efunc_add_growthterm_vect[j].append(None)
                efunc_mult_growthterm_vect[j].append(None)
            else:
                raise Exception("Currently not supporting any other node interaction types.")

        self.efunc_add_growthterm_vect = efunc_add_growthterm_vect
        self.efunc_mult_growthterm_vect = efunc_mult_growthterm_vect
        self.efunc_mult_decayterm_vect = efunc_mult_decayterm_vect

        dcdt_vect_s = []

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
            ntype = self.node_types[nde_i]  # get the node type
            # if we're not dealing with a 'signal' node that's written externally:
            if ntype is not NodeType.signal:
                dcdt_vect_s.append(r_max_s[nde_i]*efunc_mult_growthterm_vect[nde_i]*efunc_add_growthterm_vect[nde_i]
                                   - c_s[nde_i] * d_max_s[nde_i] * efunc_mult_decayterm_vect[nde_i])
            else:
                dcdt_vect_s.append(0)


        # analytical rate of change of concentration vector for the network:
        self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)

        self._include_process = False  # Set the internal boolean to True for consistency

        # Generate the optimization "energy" function as well as jacobians and hessians for the system:
        self._generate_optimization_functions()


    def build_analytical_model_with_process(self,
                                            control_edges_dict: dict|None = None,
                                            control_node_dict: dict | None = None,
                                            prob_acti: float=0.5,
                                            edge_types: list|ndarray|None=None,
                                            add_interactions: bool=True):
        '''

        '''

        self._reduced_dims = False  # always build models in full dimensions

        if edge_types is None:
            self.edge_types = self.get_edge_types(p_acti=prob_acti)

        else:
            self.edge_types = edge_types

        self.set_edge_types(edge_types, add_interactions)

        # FIXME: we ultimately want to add more than one of each node type to the network (i.e. more
        # than one effector, or multiple processes and sensors, etc)

        if control_node_dict is None: # default behaviour
            # Set indices of root and effector nodes to the highest and lowest degree nodes:
            self._root_i = self.nodes_by_out_degree[0]
            self._effector_i = self.nodes_by_out_degree[-1]
            # add sensor and process nodes to the network as new nodes:
            self._sensor_i = self.N_nodes
            self._process_i = 1 + self.N_nodes

            if control_edges_dict is None: # By default assign activator edge types
                ps_edge = EdgeType.A.value
                sr_edge = EdgeType.A.value

            else: # allow user to specify
                ps_edge = control_edges_dict['process-sensor'].value
                sr_edge = control_edges_dict['sensor-root'].value

            # Connect the new nodes to the network with new edges
            self.GG.add_edge(self._process_i, self._sensor_i,
                             edge_type=ps_edge)

            self.GG.add_edge(self._sensor_i, self._root_i,
                             edge_type=sr_edge)

            # By default the effector-process edge type must be neutral as the
            # process specifies the interaction of the effector on it:
            self.GG.add_edge(self._effector_i, self._process_i,
                             edge_type=EdgeType.N.value)

            # update the graph nodes and node properties:
            self.N_nodes = self.N_nodes + 2  # We added the sensor and process nodes to the graph
            self.nodes_index = np.arange(self.N_nodes)  # We make a new nodes list

            # Harvest data from edge attributes for edge property re-assignment:
            self.read_edge_info_from_graph()

            # Indices of key new edges:
            self.ei_process_sensor = self.edges_index.index((self._process_i, self._sensor_i))
            self.ei_sensor_root = self.edges_index.index((self._sensor_i, self._root_i))

            # Re-calculate key characteristics of the graph after adding in new nodes and edges:
            self.characterize_graph()

        else:
            self._root_i = control_node_dict['root']
            self._effector_i = control_node_dict['effector']
            self._sensor_i = control_node_dict['sensor']
            self._process_i = control_node_dict['process']

            # Indices of key new edges:
            self.ei_process_sensor = self.edges_index.index((self._process_i, self._sensor_i))
            self.ei_effector_process = self.edges_index.index((self._effector_i, self._process_i))

            # Override the edge-type for the control loop effector-process:
            self.edge_types[self.ei_effector_process] = EdgeType.N # This is always neutral

            # Update the edge types on the graph edges:
            self.set_edge_types(self.edge_types, add_interactions)

        # See if there are paths connecting the hub and effector node:
        try:
            self.root_effector_paths = sorted(nx.shortest_simple_paths(self.GG, self._root_i, self._effector_i), reverse=True)
        except:
            self.root_effector_paths = []

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_index]  # Set all nodes to the gene type

        # Add a type tag to any nodes on the path between root hub and effector:
        for path_i in self.root_effector_paths:
            for nde_i in path_i:
                node_types[nde_i] = NodeType.path

        node_types[self._root_i] = NodeType.root  # Set the most connected node to the root hub
        node_types[self._effector_i] = NodeType.effector  # Set the least out-connected node to the effector
        node_types[self._sensor_i] = NodeType.sensor  # Set the sensor node
        node_types[self._process_i] = NodeType.process  # Set the process node

        # Set node types to the graph:
        self.node_types = node_types
        self.set_node_types(node_types)

        # Build the basic edge functions:
        self.edge_funcs = []
        for et in self.edge_types:
            if et is EdgeType.A:
                self.edge_funcs.append(self.f_acti_s)
            elif et is EdgeType.I:
                self.edge_funcs.append(self.f_inhi_s)
            elif et is EdgeType.N:
                self.edge_funcs.append(self.f_neut_s)
            else:
                raise Exception("Edge type not found!")

        # Rebuild the symbolic parameter bases:
        c_s = sp.IndexedBase('c')
        K_s = sp.IndexedBase('K')
        n_s = sp.IndexedBase('n')
        r_max_s = sp.IndexedBase('r_max')
        d_max_s = sp.IndexedBase('d_max')

        # # These are needed for lambdification of analytical models:
        self.K_vect_s = [K_s[i] for i in range(self.N_edges)]
        self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
        self.r_vect_s = [r_max_s[i] for i in self.nodes_index]
        self.d_vect_s = [d_max_s[i] for i in self.nodes_index]
        self.c_vect_s = [c_s[i] for i in self.nodes_index]

        # Create the analytic equations governing the process:
        self.set_analytic_process(self.c_vect_s[self._effector_i], self.c_vect_s[self._process_i])

        # Create the edge-function collections at each node for the GRN interactions:
        efunc_vect = [[] for i in self.nodes_index]
        for ei, ((i, j), fun_type) in enumerate(zip(self.edges_index, self.edge_funcs)):
            efunc_vect[j].append(fun_type(self.c_vect_s[i], self.K_vect_s[ei], self.n_vect_s[ei]))

        # Create the time-change vector with the process node math applied:
        dcdt_vect_s = []

        for nde_i, (fval_set, ntype) in enumerate(zip(efunc_vect, node_types)):
            if ntype is NodeType.process:  # if we're dealing with the phys/chem process node...
                dcdt_vect_s.append(self.dEdt_s)  # ...append the osmotic strain rate equation.

            else:  # if it's any other kind of node insert the conventional GRN node dynamics
                if add_interactions:
                    if len(fval_set) == 0:
                        normf = 1
                    else:
                        normf = sp.Rational(1, len(fval_set))

                    dcdt_vect_s.append(self.r_vect_s[nde_i] * np.sum(fval_set) * normf -
                                       self.c_vect_s[nde_i] * self.d_vect_s[nde_i])
                else:
                    dcdt_vect_s.append(self.r_vect_s[nde_i]*np.prod(fval_set) -
                                       self.c_vect_s[nde_i]*self.d_vect_s[nde_i])


        # The last thing we need to do is add on a rate term for those nodes that have no inputs,
        # as they're otherwise ignored in the construction:
        for nde_i, di in enumerate(self.in_degree_sequence):
            if di == 0 and add_interactions is True:
                dcdt_vect_s[nde_i] += self.r_vect_s[nde_i]

        # analytical rate of change of concentration vector for the network:
        self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)

        self._include_process = True # Set the internal boolean to True for consistency
        # Generate the optimization "energy" function as well as jacobians and hessians for the system:
        self._generate_optimization_functions()


    def _make_node_edge_indices(self):
        '''

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

    def _generate_optimization_functions(self):
        '''

        '''

        if self._reduced_dims and self._solved_analytically is False:
            dcdt_vect_s = self.dcdt_vect_reduced_s
            c_vect_s = self.c_vect_reduced_s

        else:
            dcdt_vect_s = self.dcdt_vect_s
            c_vect_s = self.c_vect_s

        if self._include_process:
            lambda_params = [self.c_vect_s,
                             self.r_vect_s,
                             self.d_vect_s,
                             self.K_vect_s,
                             self.n_vect_s,
                             self.process_params_s]


            lambda_params_r = [c_vect_s,
                             self.r_vect_s,
                             self.d_vect_s,
                             self.K_vect_s,
                             self.n_vect_s,
                             self.process_params_s]

        else:
            lambda_params = [self.c_vect_s,
                             self.r_vect_s,
                             self.d_vect_s,
                             self.K_vect_s,
                             self.n_vect_s,
                             ]


            lambda_params_r = [c_vect_s,
                             self.r_vect_s,
                             self.d_vect_s,
                             self.K_vect_s,
                             self.n_vect_s]

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
                self.sol_cset_f[ci.indices[0]] = sp.lambdify(lambda_params_r, eqci)

    def reduce_model_dimensions(self):
        '''

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
                for si in self.signal_inds:
                    if si in self.c_master_i:
                        self.signal_reduced_inds.append(self.c_master_i.index(si))

                self.nonsignal_reduced_inds = np.setdiff1d(np.arange(len(self.c_master_i)), self.signal_reduced_inds)

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
                if self._include_process:
                    lambda_params_r = [self.r_vect_s,
                                       self.d_vect_s,
                                       self.K_vect_s,
                                       self.n_vect_s,
                                       self.process_params_s]

                else:
                    lambda_params_r = [self.r_vect_s,
                                       self.d_vect_s,
                                       self.K_vect_s,
                                       self.n_vect_s]

                self.sol_cset_f = {}
                for ci, eqci in self.sol_cset_s.items():
                    self.sol_cset_f[ci.indices[0]] = sp.lambdify(lambda_params_r, eqci)

            # Generate the optimization "energy" function as well as jacobians and hessians for the system.
            # self.sol_cset_s is lambdified in the following method:
            self._generate_optimization_functions()

    def f_acti_s(self, cc, kk, nn):
        '''

        '''
        return ((cc / kk) ** nn) / (1 + (cc / kk) ** nn)

    def f_inhi_s(self, cc, kk, nn):
        '''

        '''
        return 1 / (1 + (cc / kk) ** nn)

    def f_neut_s(self, cc, kk, nn):
        '''
        Calculates a "neutral" edge interaction, where
        there is neither an activation nor inhibition response.
        '''
        return 1

    def plot_3d_streamlines(self,
                            c0: ndarray,
                            c1: ndarray,
                            c2: ndarray,
                            dc0: ndarray,
                            dc1: ndarray,
                            dc2: ndarray,
                            point_data: ndarray|None = None,
                            axis_labels: list|tuple|ndarray|None=None,
                            n_points: int=100,
                            source_radius: float=0.5,
                            source_center: tuple[float, float, float]=(0.5, 0.5, 0.5),
                            tube_radius: float=0.003,
                            arrow_scale: float=1.0,
                            lighting: bool = False,
                            cmap: str = 'magma'
                            ):
        '''

        '''

        pvgrid = pv.RectilinearGrid(c0, c1, c2)  # Create a structured grid for our space

        if point_data is not None:
            pvgrid.point_data["Magnitude"] = point_data.ravel()

        if axis_labels is not None:
            labels = dict(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])
        else:
            labels = dict(xtitle='c0', ytitle='c1', ztitle='c2')

        vects_control = np.vstack((dc0.T.ravel(), dc1.T.ravel(), dc2.T.ravel())).T

        # vects_control = np.vstack((np.zeros(dndt_vect.shape), np.zeros(dndt_vect.shape), dVdt_vect/p.vol_cell_o)).T
        pvgrid["vectors"] = vects_control * 0.1
        pvgrid.set_active_vectors("vectors")

        streamlines, src = pvgrid.streamlines(vectors="vectors",
                                              return_source=True,
                                              n_points=n_points,
                                              source_radius=source_radius,
                                              source_center=source_center
                                              )

        arrows = streamlines.glyph(orient="vectors", factor=arrow_scale)

        pl = pv.Plotter()
        pl.add_mesh(streamlines.tube(radius=tube_radius), lighting=lighting, cmap=cmap)
        pl.add_mesh(arrows, cmap=cmap)
        pl.remove_scalar_bar("vectors")
        pl.remove_scalar_bar("GlyphScale")
        pl.show_grid(**labels)

        return pl

    def brute_force_phase_space(self,
                                N_pts: int=15,
                                cmin: float=0.0,
                                cmax: float|list=1.0,
                                Ki: float|list=0.5,
                                ni:float|list=10.0,
                                ri:float|list=1.0,
                                di:float|list=1.0,
                                zer_thresh: float=0.01,
                                include_signals: bool = False
                                ):
        '''

        '''

        if self.dcdt_vect_f is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        # Create parameter vectors for the model:
        self.create_parameter_vects(Ki, ni, ri, di)

        if self._reduced_dims and self._solved_analytically is False:
            N_nodes = len(self.c_vect_reduced_s)
            dcdt_vect_f = self.dcdt_vect_reduced_f
            c_vect_s = self.c_vect_reduced_s
        else:
            N_nodes = self.N_nodes
            dcdt_vect_f = self.dcdt_vect_f
            c_vect_s = self.c_vect_s

        c_vect_set, C_M_SET, c_lin_set = self._generate_state_space(c_vect_s,
                                                   Ns=N_pts,
                                                   cmin=cmin,
                                                   cmax=cmax,
                                                   include_signals=include_signals)

        M_shape = C_M_SET[0].shape

        dcdt_M = np.zeros(c_vect_set.shape)

        for i, c_vecti in enumerate(c_vect_set):
            if self._include_process is False:
                dcdt_i = dcdt_vect_f(c_vecti,
                                          self.r_vect,
                                          self.d_vect,
                                          self.K_vect,
                                          self.n_vect)
            else:
                dcdt_i = dcdt_vect_f(c_vecti,
                                          self.r_vect,
                                          self.d_vect,
                                          self.K_vect,
                                          self.n_vect,
                                          self.process_params_f)
            dcdt_M[i] = dcdt_i * 1

        dcdt_M_set = []
        for dci in dcdt_M.T:
            dcdt_M_set.append(dci.reshape(M_shape))

        self.c_lin_set = c_lin_set
        self.C_M_SET = C_M_SET
        self.M_shape = M_shape

        self.dcdt_M_set = np.asarray(dcdt_M_set)
        self.dcdt_dmag = np.sqrt(np.sum(self.dcdt_M_set ** 2, axis=0))
        self.dcdt_zeros = ((self.dcdt_dmag / self.dcdt_dmag.max()) < zer_thresh).nonzero()

        return self.dcdt_zeros, self.dcdt_M_set, self.dcdt_dmag, self.c_lin_set, self.C_M_SET

    def create_parameter_vects(self, Ki: float|list=0.5,
                                ni:float|list=3.0,
                                ri:float|list=1.0,
                                di:float|list=1.0
                                ):
        '''

        '''
        # Create parameter vectors as the same parameters for all edges and nodes in the network:
        if type(Ki) is not list:
            K_vect = []
            for ei in range(self.N_edges):
                K_vect.append(Ki)

        else:
            K_vect = Ki

        if type(ni) is not list:
            n_vect = []
            for ei in range(self.N_edges):
                n_vect.append(ni)
        else:
            n_vect = ni

        if type(ri) is not list:
            r_vect = []
            for nde_i in range(self.N_nodes):
                r_vect.append(ri)
        else:
            r_vect = ri

        if type(di) is not list:
            d_vect = []
            for nde_i in range(self.N_nodes):
                d_vect.append(di)
        else:
            d_vect = di

        self.K_vect = K_vect
        self.n_vect = n_vect
        self.r_vect = r_vect
        self.d_vect = d_vect

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

    def _generate_state_space(self,
                              c_vect_s: list|ndarray,
                              Ns: int=3,
                              cmin: float=0.0,
                              cmax: float=1.0,
                              include_signals: bool = False # include signal node states in the search?
                              ):
        '''
        
        '''
        c_test_lin_set = []

        if self._reduced_dims and self._solved_analytically is False:
            signal_inds = self.signal_reduced_inds
            nonsignal_inds = self.nonsignal_reduced_inds
        else:
            signal_inds = self.signal_inds
            nonsignal_inds = self.nonsignal_inds

        if include_signals is False:
            # Create a c_vect sampled to the non-signal nodes:
            c_vect = np.asarray(c_vect_s)[nonsignal_inds].tolist()

        else:
            c_vect = c_vect_s # otherwise use the whole vector

        for nd_i, ci in enumerate(c_vect):
                i = c_vect.index(ci)
                if self._include_process is True and i == self._process_i:
                    c_test_lin_set.append(np.linspace(self.Vp_min, self.Vp_max, Ns))
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
                                     Ki: float | list = 0.5,
                                     ni: float | list = 3.0,
                                     ri: float | list = 1.0,
                                     di: float | list = 1.0,
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

        # Create parameter vectors for the model:
        self.create_parameter_vects(Ki, ni, ri, di)

        # Determine the set of additional arguments to the optimization function:
        if self._include_process is False:
            function_args = (self.r_vect, self.d_vect, self.K_vect, self.n_vect)
        else:
            function_args = (self.r_vect, self.d_vect, self.K_vect, self.n_vect,
                             self.process_params_f)

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
            c_test_set, _, _ = self._generate_state_space(c_vect_s,
                                                       Ns=Ns,
                                                       cmin=cmin,
                                                       cmax=cmax,
                                                       include_signals=include_signals)

            if c_bounds is None:
                c_bounds = [(cmin, cmax) for i in range(N_nodes)]

                if self._include_process:
                    if self._reduced_dims is False:
                        c_bounds[self._process_i] = (self.Vp_min, self.Vp_max)
                    else:
                        if self.c_vect_s[self._process_i] in self.c_vect_reduced_s:
                            i = self.c_vect_reduced_s.index(self.c_vect_s[self._process_i])
                            c_bounds[i] = (self.Vp_min, self.Vp_max)

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
                    if self._include_process is False: # If we're not using the process, constrain all concs to be above zero
                        if (np.all(np.asarray(sol_root) >= 0.0)):
                            mins_found.append(sol_root)
                    else:
                        # get the nodes that must be constrained above zero:
                        conc_nodes = np.setdiff1d(self.nodes_index, self._process_i)
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

            if self._include_process is False:
                func_args = (cmins, self.r_vect, self.d_vect, self.K_vect, self.n_vect)

            else:
                func_args = (cmins, self.r_vect, self.d_vect, self.K_vect, self.n_vect,
                             self.process_params_f)

            # print(f'dcdt at min: {self.dcdt_vect_f(cmins, r_vect, d_vect, K_vect, n_vect)}')

            solution_dict['Change at Minima'] = self.dcdt_vect_f(*func_args)

            jac = self.jac_f(*func_args)

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


    def set_analytic_process(self, c_effector: Symbol|Indexed, c_process: Symbol|Indexed):
        '''

        c_effector : Symbol
            Symbolic concentration from a node in the GRN network that represents the moles of
            osmolyte inside the cell.

        '''
        # FIXME: we need to put in the c_factors here for the environmental osmolytes

        # Defining analytic equations for an osmotic cell volume change process:
        A_s, R_s, T_s, ni_s, m_s, V_s, Vc_s, dm_s, mu_s, Y_s, r_s = sp.symbols('A, R, T, ni, m, V, V_c, d_m, mu, Y, r',
                                                                              real=True)
        # Normalized parameters:
        Ap_s, mp_s, Ac_s, nc_s, mc_s, epsilon_s = sp.symbols('A_p, m_p, A_c, n_c, m_c, epsilon', real=True)

        dVdt_0_s = A_s ** 2 * R_s * T_s * (ni_s - m_s * V_s) / (8 * dm_s * mu_s * V_s)
        dVdt_1_s = (A_s ** 2 / (8 * dm_s * mu_s)) * (
                    R_s * T_s * ((ni_s / V_s) - m_s) - sp.Rational(4, 3) * ((Y_s * dm_s * (V_s - Vc_s) / (r_s * Vc_s))))

        # Rate of change of Vp with respect to time for Vp < 1.0 is:
        dVpdt_0_s = (dVdt_0_s.subs(
            [(V_s, c_process * Vc_s), (A_s, Ap_s * Ac_s), (ni_s, nc_s * c_effector), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Rate of change of Vp with respect to time for Vp >= 1.0
        dVpdt_1_s = (dVdt_1_s.subs(
            [(V_s, c_process * Vc_s), (A_s, Ap_s * Ac_s), (ni_s, nc_s * c_effector), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Volume change rates (which are the input into the sensor node) are:
        # FIXME: Substitute in expression for strain in terms of Vp here!
        dEdt_0_s = dVpdt_0_s
        dEdt_1_s = dVpdt_1_s

        # Piecewise function that defines this normalized-parameter osmotic cell volume change problem
        # as a strain rate:
        self.dEdt_s = sp.Piecewise((dEdt_0_s, c_process < 1.0), (dEdt_1_s, True))

        # Go ahead and initialize some parameters for this process function: # FIXME these need to be
        # made easier to input, vary and change:
        self.m_f = 0.8  # Normalized environmental osmolyte concentration (high)
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

        self.Vp_min = 0.2 # minimum relative volume that can be achieved
        self.Vp_max = 2.0 # maximum relative volume that can be achieved

        # symbolic parameters for the dV/dt process (these must be augmented onto the GRN parameters
        # when lambdifying):
        self.process_params_s = (mp_s, Ap_s, Vc_s, nc_s, mc_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s)

        self.dEdt_f = sp.lambdify([c_process, c_effector, self.process_params_s], self.dEdt_s)

        # Numeric parameters for the dV/dt process (these must be augmented onto the GRN parameters
        # when using numerical network equations):
        self.process_params_f = (
                               self.m_f,
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

    def set_node_type_path(self,
                           root_i: int|None = None,
                           effector_i: int|None=None):
        '''

        '''

        # Set indices of root and effector nodes to the highest and lowest degree nodes when none are specified:
        if root_i is None:
            root_i = self.nodes_by_out_degree[0]

        if effector_i is None:
            effector_i = self.nodes_by_out_degree[-1]

        # See if there are paths connecting the hub and effector node:
        try:
            paths_i = sorted(nx.shortest_simple_paths(self.GG, root_i, effector_i), reverse=True)
        except:
            paths_i = []

        node_types = [NodeType.gene for i in self.nodes_index]  # Set all nodes to the gene type
        # Add any nodes on the path:
        for path_i in paths_i:
            for nde_i in path_i:
                node_types[nde_i] = NodeType.path
        # Add the path endpoint node highlights:
        node_types[root_i] = NodeType.root
        node_types[effector_i] = NodeType.effector

        # Set node types to the graph:
        self.node_types = node_types
        self.set_node_types(node_types)

    def multistability_search(self,
                              N_multi: int,
                              tol: float=1.0e-3,
                              N_iter:int =100,
                              verbose: bool=True,
                              N_space: int=3,
                              N_round_sol: int=6,
                              N_round_unique_sol: int=1,
                              Ki: float | list = 0.5,
                              ni: float | list = 3.0,
                              di: float | list = 1.0,
                              search_tol: float = 1.0e-15,
                              add_interactions: bool = True,
                              unique_sols: bool = True
                              ):
        '''
        By randomly generating sets of edge interaction (i.e. activator or inhibitor), find
        as many unique multistable systems as possible for a given base network.

        This does not redefine any default 'edge_types' that have been assigned to the network.

        '''

        multisols = []
        multisol_edges = []
        numsol_list = []

        for i in range(N_iter):
            edge_types = self.get_edge_types(p_acti=0.5)
            self.build_analytical_model(edge_types=edge_types,
                                        add_interactions=add_interactions)
            sols_0 = self.optimized_phase_space_search(Ns=N_space,
                                                       cmax=1.5*np.max(self.in_degree_sequence),
                                                       round_sol=N_round_sol,
                                                       Ki = Ki,
                                                       di = di,
                                                       ni = ni,
                                                       tol=search_tol,
                                                       method="Root"
                                                       )

            solsM = self.find_attractor_sols(sols_0,
                                             tol=tol,
                                             unique_sols=unique_sols,
                                             verbose=False,
                                             N_round=N_round_unique_sol)

            if len(solsM):
                num_sols = solsM.shape[1]
            else:
                num_sols = 0

            if num_sols >= N_multi:
                edge_types_l = edge_types.tolist()
                if edge_types_l not in multisol_edges: # If we don't already have this combo:
                    if verbose:
                        print(f'Found solution with {num_sols} states on iteration {i}')
                    multisols.append([sols_0, edge_types])
                    numsol_list.append(num_sols)
                    multisol_edges.append(edge_types_l)

        return numsol_list, multisols

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

        for ti, tt in enumerate(tvect):
            dcdt = self.dcdt_vect_f(cvecti, self.r_vect, self.d_vect, self.K_vect, self.n_vect)
            cvecti += dt * dcdt

            if c_signals is not None:
                # manually set the signal node values:
                cvecti[self.signal_inds] = c_signals[ti, self.signal_inds]

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


    def find_state_match(self, solsM: ndarray, cvecti: list|ndarray):
        '''


        '''

        # now what we need is a pattern match from concentrations to the stable states:
        errM = []
        for soli in solsM.T:
            sdiff = soli - cvecti
            errM.append(np.sum(sdiff ** 2))
        errM = np.asarray(errM)
        state_best_match = (errM == errM.min()).nonzero()[0][0]

        return state_best_match, errM[state_best_match]


    def create_transition_network(self,
                                  solsM: ndarray,
                                  dt: float=1.0e-3,
                                  tend: float=100.0,
                                  sigtstart: float=33.0,
                                  sigtend: float=66.0,
                                  sigmax: float=2.0,
                                  delta_window: float=1.0,
                                  verbose: bool=True,
                                  tol: float=1.0e-6,

                                  ):
        '''


        '''

        # See if we can build a transition matrix/diagram by starting the system
        # in different states and seeing which state it ends up in:

        c_zeros = np.zeros(self.N_nodes)  # start everything out low

        # Create a steady-state solutions matrix that is stacked with the
        # 'zero' or 'base' state:
        solsM_with0 = np.column_stack((c_zeros, solsM))

        # stateM = np.zeros((solsM.shape[1], solsM.shape[1]))
        G_states = nx.DiGraph()

        max_states = solsM_with0.T.shape[1]

        for stateio, cvecto in enumerate(solsM_with0.T):  # start the system off in each of the states

            # For each signal in the network:
            for si, sigi in enumerate(self.signal_inds):
                cvecti = cvecto.copy()  # reset the state to the desired starting state
                sig_inds = [sigi]
                sig_times = [(sigtstart, sigtend)]
                sig_mags = [(0.0, sigmax)]

                # Run the time simulation:
                concs_time, tvect = self.run_time_sim(tend,
                                                      dt,
                                                      cvecti,
                                                      sig_inds,
                                                      sig_times,
                                                      sig_mags,
                                                      dt_samp=0.15)

                it_30low = (tvect <= sigtstart - delta_window).nonzero()[0]
                it_30high = (tvect >= sigtstart -2*delta_window).nonzero()[0]
                it_30 = np.intersect1d(it_30low, it_30high)[0]

                concs_before = concs_time[it_30, :]
                concs_after = concs_time[-1, :]

                if stateio == 0 and si == 0:  # If we're on the zeros vector we've transitioned from {0} to some new state:
                    statejo, errio = self.find_state_match(solsM_with0, concs_before)
                    if errio < tol:
                        G_states.add_edge(0, statejo, transition=-1)

                    else: # otherwise it's not a match so add the new state to the system:
                        solsM_with0 = np.column_stack((solsM_with0, concs_before))
                        statejo, errio = self.find_state_match(solsM_with0, concs_before)
                        G_states.add_edge(0, statejo, transition=-1)

                    if verbose:
                        print(f'From state 0 spontaneously to state {statejo}, error{errio}')

                statei, erri = self.find_state_match(solsM_with0, concs_before)
                statej, errj = self.find_state_match(solsM_with0, concs_after)

                if erri > tol:
                    solsM_with0 = np.column_stack((solsM_with0, concs_before))
                    statei, erri = self.find_state_match(solsM_with0, concs_before)

                if errj > tol:
                    solsM_with0 = np.column_stack((solsM_with0, concs_after))
                    statej, errj = self.find_state_match(solsM_with0, concs_after)

                G_states.add_edge(statei, statej, transition=si)

                if verbose:
                    print(f'From state {statei} with signal {si} to state {statej}, errors{erri, errj}')

        return G_states

    def plot_transition_network(self, G_states: DiGraph, solsM: ndarray, save_graph_net: str):
        '''

        '''


        edgedata_Gstates = nx.get_edge_attributes(G_states, "transition")
        nodes_Gstates = list(G_states.nodes)

        # cmap = colormaps['magma']
        cmap = colormaps['rainbow_r']
        norm = colors.Normalize(vmin=0, vmax=solsM.shape[1])

        G = pgv.AGraph(strict=False,
                       splines=True,
                       directed=True,
                       concentrate=False,
                       # nodesep=0.1,
                       # ranksep=0.3,
                       dpi=300)

        for ni, nde_stateo in enumerate(nodes_Gstates):
            nde_color = colors.rgb2hex(cmap(norm(ni)))
            nde_color += '80'  # add some transparancy to the node
            # if norm(ni) >= 0.6:
            #     nde_font_color = 'Black'
            # else:
            #     nde_font_color = 'White'
            nde_font_color = 'Black'

            nde_state = f'State {nde_stateo}'

            G.add_node(nde_state,
                       style='filled',
                       fillcolor=nde_color,
                       fontcolor=nde_font_color,
                       # fontname=net_font_name,
                       # fontsize=node_font_size,
                       )

        for (eio, ejo), etranso in edgedata_Gstates.items():
            ei = f'State {eio}'
            ej = f'State {ejo}'
            if etranso == -1:
                etrans = 'Spont.'
            else:
                etrans = f'S{etranso}'
            G.add_edge(ei, ej, label=etrans)

        G.layout(prog='dot')  # default to dot

        G.draw(save_graph_net)


    def plot_sols_array(self, solsM: ndarray, figsave: str | None = None, cmap: str | None =None):
        '''

        '''

        if cmap is None:
            cmap = 'magma'

        state_labels = [f'State {i +1}' for i in range(solsM.shape[1])]
        gene_labels = np.asarray(self.nodes_list)[self.nonsignal_inds]
        fig, ax = plt.subplots()
        im = ax.imshow(solsM[self.nonsignal_inds, :], cmap=cmap)
        # plt.colorbar(label='Expression Level')
        ax.set_xticks(np.arange(len(state_labels)), labels=state_labels)
        ax.set_yticks(np.arange(len(gene_labels)), labels=gene_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label='Expression Level')

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format='png')

        return fig, ax

    def plot_knockout_arrays(self, knockout_sol_set: list|ndarray, figsave: str=None):
        '''
        Plot all steady-state solution arrays in a knockout experiment solution set.

        '''

        # let's plot this as a multidimensional set of master arrays:
        knock_flat = []
        for kmat in knockout_sol_set:
            for ki in kmat:
                knock_flat.extend(ki)

        vmax = np.max(knock_flat)
        vmin = np.min(knock_flat)

        cmap = 'magma'

        N_axis = len(knockout_sol_set)

        fig, axes = plt.subplots(1, N_axis, sharey=True, sharex=True)

        for i, (axi, solsMio) in enumerate(zip(axes, knockout_sol_set)):
            if len(solsMio):
                solsMi = solsMio
            else:
                solsMi = np.asarray([np.zeros(self.N_nodes)]).T
            axi.imshow(solsMi, aspect="equal", vmax=vmax, vmin=vmin, cmap=cmap)
            axi.axis('off')
            if i != 0:
                axi.set_title(f'c{i - 1}')
            else:
                axi.set_title(f'Full')

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format='png')

        return fig, axes

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


    def param_space_search(self,
                           N_pts: int=3,
                           ri: float = 1.0,
                           ni: float = 3.0,
                           K_min: float = 0.1,
                           K_max: float = 2.0,
                           d_min: float = 0.5,
                           d_max: float = 10.0,
                           sol_round: int = 1,
                           N_search: int = 3,
                           search_round_sol: int=6,
                           tol: float=1.0e-3,
                           cmax_multi: float=2.0,
                           verbose: bool=True,
                           hold_d_const: bool=True,
                           di: float=0.5):
        '''
        Search parameter space of a model to find paameter combinations that give different multistable
        states.
        '''

        if self._reduced_dims and self._solved_analytically is False:
            N_nodes = len(self.c_vect_reduced_s)

        else:
            N_nodes = self.N_nodes

        # What we wish to create is a parameter space search, as this net is small enough to enable that.
        Klin = np.linspace(K_min, K_max, N_pts)
        dlin = np.linspace(d_min, d_max, N_pts)

        param_lin_set = []

        for edj_i in range(self.N_edges):
            param_lin_set.append(Klin*1) # append the linear K-vector choices for each edge

        if hold_d_const is False:
            for nde_i in range(N_nodes):
                param_lin_set.append(dlin*1)

        # Create a set of matrices specifying the concentration grid for each
        # node of the network:
        param_M_SET = np.meshgrid(*param_lin_set, indexing='ij')

        # Create linearized arrays for each concentration, stacked into one column per node:
        param_test_set = np.asarray([pM.ravel() for pM in param_M_SET]).T

        bif_space_M = [] # Matrix holding the parameter values and number of unique stable solutions
        sols_space_M = []

        if verbose:
            print(param_M_SET[0].ravel().shape)

        if verbose:
            print(f'Search cmax will be {cmax_multi * np.max(self.in_degree_sequence)}')

        for param_set_i in param_test_set:
            kvecti = param_set_i[0:self.N_edges].tolist()

            if hold_d_const is False:
                dvecti = param_set_i[self.N_edges:].tolist()
            else:
                dvecti = di

            self.create_parameter_vects(Ki=kvecti, ni=ni, ri=ri, di=dvecti)

            sols_0 = self.optimized_phase_space_search(Ns=N_search,
                                                       cmax=cmax_multi * np.max(self.in_degree_sequence),
                                                       round_sol=search_round_sol,
                                                       Ki=self.K_vect,
                                                       di=self.d_vect,
                                                       ni=self.n_vect,
                                                       tol=1.0e-6,
                                                       method="Root"
                                                       )

            solsM = self.find_attractor_sols(sols_0, tol=tol, verbose=False, unique_sols=True, sol_round=sol_round)

            if len(solsM):
                num_sols = solsM.shape[1]
            else:
                num_sols = 0

            bif_space_M.append([*self.K_vect, *self.d_vect, num_sols])
            sols_space_M.append(solsM)

        return np.asarray(bif_space_M), sols_space_M

    def save_model_equations(self,
                             save_eqn_image: str,
                             save_reduced_eqn_image: str,
                             save_eqn_csv: str
                             ):
        '''

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

        if self.dcdt_vect_reduced_s is not None:
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

        with open(save_eqn_csv, 'w', newline="") as file:
            csvwriter = csv.writer(file)  # 2. create a csvwriter object
            csvwriter.writerow(header)  # 4. write the header
            csvwriter.writerows(eqns_to_write)  # 5. write the rest of the data


    def gene_knockout_experiment(self,
                                 Ns: int=3,
                                 cmin: float=0.0,
                                 cmax: float=1.0,
                                 Ki: float | list = 0.5,
                                 ni: float | list = 3.0,
                                 ri: float | list = 1.0,
                                 di: float | list = 1.0,
                                 tol:float = 1.0e-6,
                                 round_sol: int=6,
                                 round_unique_sol: int=2,
                                 unique_sols: bool=True,
                                 sol_tol: float = 1.0e-3,
                                 verbose: bool = True,
                                 save_file_basename: str | None = None
                                 ):
        '''
        Performs a knockout of all genes in the network, computing all possible steady-state
        solutions for the resulting knockout. This is different from the transition matrix,
        as the knockouts aren't a temporary perturbation, but a long-term silencing.

        '''


        if self.dcdt_vect_s is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        knockout_sol_set = []
        # knockout_dcdt_s_set = []
        # knockout_dcdt_f_set = []

        if save_file_basename is not None:
            save_file_list = [f'{save_file_basename}_allc.csv']
            save_file_list.extend([f'{save_file_basename}_ko_c{i}.csv' for i in range(self.N_nodes)])

        else:
            save_file_list = [None]
            save_file_list.extend([None for i in range(self.N_nodes)])

        # Create parameter vectors for the model:
        self.create_parameter_vects(Ki, ni, ri, di)

        # Solve the system with all concentrations:
        sols_0 = self.optimized_phase_space_search(Ns=Ns,
                                                   cmax=cmax,
                                                   round_sol=round_sol,
                                                   Ki=self.K_vect,
                                                   di=self.d_vect,
                                                   ni=self.n_vect,
                                                   tol=tol,
                                                   method="Root"
                                                   )

        # Screen only for attractor solutions:
        solsM = self.find_attractor_sols(sols_0,
                                         tol=sol_tol,
                                         verbose=verbose,
                                         unique_sols=unique_sols,
                                         sol_round=round_unique_sol,
                                         save_file=save_file_list[0])

        if verbose:
            print(f'-------------------')


        knockout_sol_set.append(solsM.copy())

        for i, c_ko_s in enumerate(self.c_vect_s): # Step through each concentration

            # Define a new change vector by substituting in the knockout value for the gene (c=0) and
            # clamping the gene at that level by setting its change rate to zero:
            dcdt_vect_ko_s = self.dcdt_vect_s.copy() # make a copy of the change vector
            # dcdt_vect_ko_s = dcdt_vect_ko_s.subs(c_ko_s, 0)
            dcdt_vect_ko_s.row_del(i) # Now we have to remove the row for this concentration

            # create a new symbolic concentration vector that has the silenced gene removed:
            c_vect_ko = self.c_vect_s.copy()
            c_vect_ko.remove(c_ko_s)

            # do a similar thing for conc. indices so we can reconstruct solutions easily:
            nodes_ko = self.nodes_index.copy()
            del nodes_ko[i] # delete the ith index

            # knockout_dcdt_s_set.append(dcdt_vect_ko_s) # Store for later

            if self._include_process is False:
                lambda_params = [c_vect_ko,
                                 c_ko_s,
                                 self.r_vect_s,
                                 self.d_vect_s,
                                 self.K_vect_s,
                                 self.n_vect_s,
                                 ]

            else:
                lambda_params = [c_vect_ko,
                                 c_ko_s,
                                 self.r_vect_s,
                                 self.d_vect_s,
                                 self.K_vect_s,
                                 self.n_vect_s,
                                 self.process_params_s
                                 ]

            flatten_f = np.asarray([fs for fs in dcdt_vect_ko_s])
            dcdt_vect_ko_f = sp.lambdify(lambda_params, flatten_f)

            # knockout_dcdt_f_set.append(dcdt_vect_ko_f) # store for later use

            # Determine the set of additional arguments to the optimization function -- these are different each
            # time as the clamped concentration becomes an additional known parameter:
            if self._include_process is False:
                function_args = (0.0, self.r_vect, self.d_vect, self.K_vect, self.n_vect)
            else:
                function_args = (0.0, self.r_vect, self.d_vect, self.K_vect, self.n_vect,
                                 self.process_params_f)

            # Generate the points in state space to sample at:
            c_test_set, _, _ = self._generate_state_space(c_vect_ko,
                                                       Ns=Ns,
                                                       cmin=cmin,
                                                       cmax=cmax,
                                                       include_signals=True)

            # Initialize the equillibrium point solutions to be a set:
            mins_found = []

            for c_vecti in c_test_set:

                sol_rooto = fsolve(dcdt_vect_ko_f, c_vecti, args=function_args, xtol=tol)

                # reconstruct a full-length concentration vector:
                sol_root = np.zeros(self.N_nodes)
                # the solution is defined at the remaining nodes; the unspecified value is the silenced gene
                sol_root[nodes_ko] = sol_rooto

                if self._include_process is False:  # If we're not using the process, constrain all concs to be above zero
                    if (np.all(np.asarray(sol_root) >= 0.0)):
                        mins_found.append(sol_root)
                else:
                    # get the nodes that must be constrained above zero:
                    conc_nodes = np.setdiff1d(self.nodes_index, self._process_i)
                    # Then, only the nodes that are gene products must be above zero
                    if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
                        mins_found.append(sol_root)

                mins_found = np.round(mins_found, round_sol)
                mins_found = np.unique(mins_found, axis=0).tolist()

            if verbose:
                print(f'Steady-state solutions for {self.c_vect_s[i].name} knockout:')

            # Screen only for attractor solutions:
            solsM = self.find_attractor_sols(mins_found,
                                             tol=sol_tol,
                                             verbose=verbose,
                                             unique_sols=unique_sols,
                                             sol_round=round_unique_sol,
                                             save_file=save_file_list[i + 1])

            if verbose:
                print(f'-------------------')

            knockout_sol_set.append(solsM.copy())

        # merge this into a master matrix:
        ko_M = None
        for i, ko_aro in enumerate(knockout_sol_set):
            if len(ko_aro) == 0:
                ko_ar = np.asarray([np.zeros(self.N_nodes)]).T
            else:
                ko_ar = ko_aro

            if i == 0:
                ko_M = ko_ar
            else:
                ko_M = np.hstack((ko_M, ko_ar))

        return knockout_sol_set, ko_M

    def gene_knockout_solve(self, verbose: bool = True):
        '''
        Performs a knockout of all genes in the network, attempting to analytically solve for the
        resulting knockout change vector at steady-state.

        '''

        knockout_dcdt_reduced_s_set = []
        knockout_c_reduced_s_set = []
        knockout_sol_s_set = [] # for full analytical solution equations

        if self.dcdt_vect_s is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        for i, c_ko_s in enumerate(self.c_vect_s): # Step through each concentration

            # Define a new change vector by substituting in the knockout value for the gene (c=0) and
            # clamping the gene at that level by setting its change rate to zero:
            dcdt_vect_ko_s = self.dcdt_vect_s.subs(c_ko_s, 0)
            dcdt_vect_ko_s[i] = 0

            nosol = False

            try:
                sol_csetoo = sp.nonlinsolve(dcdt_vect_ko_s, self.c_vect_s)
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

                    sol_eqn_vect = []
                    for ci, eqi in sol_cset.items():
                        sol_eqn = sp.Eq(ci, eqi)
                        sol_eqn_vect.append(sol_eqn)

                    knockout_sol_s_set.append(sol_eqn_vect) # append the expressions for the system solution

                    master_eq_list = []  # master equations to be numerically optimized (reduced dimension network equations)
                    c_vect_reduced = []  # concentrations involved in the master equations

                    if len(c_master_i):
                        for ii in c_master_i:
                            # substitute in the expressions in terms of master concentrations to form the master equations:
                            ci_solve_eq = dcdt_vect_ko_s[ii].subs([(k, v) for k, v in sol_cset.items()])
                            master_eq_list.append(ci_solve_eq)
                            c_vect_reduced.append(self.c_vect_s[ii])

                    else:  # if there's nothing in c_master_i but there are solutions in sol_cseto, then it's been fully solved:
                        if verbose:
                            print("Solution solved analytically!")

                    knockout_dcdt_reduced_s_set.append([])
                    knockout_c_reduced_s_set.append([])

                else:
                    nosol = True

            except:
                nosol = True

            if nosol:
                knockout_dcdt_reduced_s_set.append(dcdt_vect_ko_s)
                knockout_c_reduced_s_set.append(self.c_vect_s)

        return knockout_sol_s_set, knockout_dcdt_reduced_s_set, knockout_c_reduced_s_set


