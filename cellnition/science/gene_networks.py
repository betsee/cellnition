#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module
'''

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
import networkx as nx
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.enumerations import EdgeType, GraphType, NodeType
from cellnition.science.stability import Solution
import pygraphviz as pgv
import pyvista as pv

# TODO: Time simulations
# TODO: Parameter scaling module: scale K and d by 's' and apply rate scaling 'v'
# TODO: Optimization with substitution-based constraints, optimizing for parameters
# TODO: Allow multiple process to be added to the network (node with different physics)
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
                 delta: float=0.0,
                 p_edge: float=0.5):
        '''

        '''
        self.N_nodes = N_nodes # number of nodes in the network

        # Depending on whether edges are supplied by user, generate
        # a graph:

        if edges is None:
            self.generate_network(beta=beta,
                                  gamma=gamma,
                                  graph_type=graph_type,
                                  delta_in=delta,
                                  delta_out=delta,
                                  p_edge=p_edge)

            self.edges_index = self.edges_list
            self.nodes_index = self.nodes_list

        else:
            self.edges_list = edges
            self.GG = nx.DiGraph(self.edges_list)
            self.N_edges = len(self.edges_list)
            self.nodes_list = sorted(self.GG.nodes())

            self._make_node_edge_indices()

        # Calculate key characteristics of the graph
        self.characterize_graph()


        self._reduced_dims = False # Indicate that model is full dimensions
        self._include_process = False # Indicate that model does not include the process by default
        self._solved_analytically = False # Indicate that the model does not have an analytical solution

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
        # Degree analysis:
        self.in_degree_sequence = [di for ni, di in
                                   self.GG.in_degree(self.nodes_list)] # aligns with node order

        self.in_dmax = np.max(self.in_degree_sequence)


        self.out_degree_sequence = [di for ni, di in
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

    def get_edge_types(self, p_acti: float=0.5):
        '''
        Automatically generate a conse
        rved edge-type vector for use in
        model building.
        '''

        p_inhi = 1.0 - p_acti

        edge_types_o = [EdgeType.A, EdgeType.I]
        edge_prob = [p_acti, p_inhi]
        edge_types = np.random.choice(edge_types_o, self.N_edges, p=edge_prob)

        return edge_types

    def set_edge_types(self, edge_types: list|ndarray):
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
        for et in self.edge_types:
            if et is EdgeType.A:
                self.edge_funcs.append(self.f_acti_s)
            elif et is EdgeType.I:
                self.edge_funcs.append(self.f_inhi_s)
            elif et is EdgeType.N:
                self.edge_funcs.append(self.f_neut_s)

    def set_node_types(self, node_types: list|ndarray):
        '''
        Assign node types to the graph.
        '''
        self.node_types = node_types
        # Set node type as graph node attribute:
        node_attr_dict = {}
        for ni, nt in zip(self.nodes_index, node_types):
            node_attr_dict[ni] = {"node_type": nt.value}

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
                               add_interactions: bool=False):
        '''

        '''

        self._reduced_dims = False # always build models in full dimensions

        if edge_types is None:
            self.edge_types = self.get_edge_types(p_acti=prob_acti)

        else:
            self.edge_types = edge_types

        self.set_edge_types(self.edge_types)

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_index]  # Set all nodes to the gene type

        # Set node types to the graph:
        self.node_types = node_types
        self.set_node_types(node_types)

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

        efunc_vect = [[] for i in self.nodes_index]
        for ei, ((i, j), fun_type) in enumerate(zip(self.edges_index, self.edge_funcs)):
            efunc_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))

        dcdt_vect_s = []

        for ni, fval_set in enumerate(efunc_vect):
            if add_interactions:
                # if len(fval_set) == 0:
                #     normf = 1
                # else:
                #     normf = sp.Rational(1, len(fval_set))

                dcdt_vect_s.append(r_max_s[ni] * np.sum(fval_set) - c_s[ni] * d_max_s[ni])
            else:
                dcdt_vect_s.append(r_max_s[ni] * np.prod(fval_set) - c_s[ni] * d_max_s[ni])

        # The last thing we need to do is add on a rate term for those nodes that have no inputs,
        # as they're otherwise ignored in the construction:
        for ni, di in enumerate(self.in_degree_sequence):
            if di == 0 and add_interactions is True:
                dcdt_vect_s[ni] += self.r_vect_s[ni]

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

        self.set_edge_types(edge_types)

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
            self.set_edge_types(self.edge_types)

        # See if there are paths connecting the hub and effector node:
        try:
            self.root_effector_paths = sorted(nx.shortest_simple_paths(self.GG, self._root_i, self._effector_i), reverse=True)
        except:
            self.root_effector_paths = []

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_index]  # Set all nodes to the gene type

        # Add a type tag to any nodes on the path between root hub and effector:
        for path_i in self.root_effector_paths:
            for ni in path_i:
                node_types[ni] = NodeType.path

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

        for ni, (fval_set, ntype) in enumerate(zip(efunc_vect, node_types)):
            if ntype is NodeType.process:  # if we're dealing with the phys/chem process node...
                dcdt_vect_s.append(self.dEdt_s)  # ...append the osmotic strain rate equation.

            else:  # if it's any other kind of node insert the conventional GRN node dynamics
                if add_interactions:
                    if len(fval_set) == 0:
                        normf = 1
                    else:
                        normf = sp.Rational(1, len(fval_set))

                    dcdt_vect_s.append(self.r_vect_s[ni] * np.sum(fval_set) * normf - self.c_vect_s[ni] * self.d_vect_s[ni])
                else:
                    dcdt_vect_s.append(self.r_vect_s[ni] * np.prod(fval_set) - self.c_vect_s[ni] * self.d_vect_s[ni])


        # The last thing we need to do is add on a rate term for those nodes that have no inputs,
        # as they're otherwise ignored in the construction:
        for ni, di in enumerate(self.in_degree_sequence):
            if di == 0 and add_interactions is True:
                dcdt_vect_s[ni] += self.r_vect_s[ni]

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
        for ni, nn in enumerate(self.nodes_list):
            self.nodes_index.append(ni)

        self.edges_index = []
        for ei, (nni, nnj) in enumerate(self.edges_list):
            ni = self.nodes_list.index(nni)
            nj = self.nodes_list.index(nnj)
            self.edges_index.append((ni, nj))
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

            else:
                # Set most reduced system attributes to None:
                self.dcdt_vect_reduced_s = None
                self.c_vect_reduced_s = None
                self.c_master_i = None
                self.c_remainder_i = None
                self.c_vect_remainder_s = None

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
                                ):
        '''

        '''

        # Create parameter vectors for the model:
        self.create_parameter_vects(Ki, ni, ri, di)

        if self._reduced_dims and self._solved_analytically is False:
            N_nodes = len(self.c_vect_reduced_s)
            dcdt_vect_f = self.dcdt_vect_reduced_f
        else:
            N_nodes = self.N_nodes
            dcdt_vect_f = self.dcdt_vect_f

        # Create linear set of concentrations over the desired range
        # for each node of the network:
        c_lin_set = []
        if type(cmax) is list: # Allow for different maxima along each axis of the space:
            for i, cmi in zip(range(N_nodes), cmax):
                if self._reduced_dims is False and self._include_process and i == self._process_i:
                    c_lin_set.append(np.linspace(self.Vp_min, self.Vp_max, N_pts))
                else:
                    c_lin_set.append(np.linspace(cmin, cmi, N_pts))

        else:
            for i in range(N_nodes):
                if self._reduced_dims is False and self._include_process and i == self._process_i:
                    c_lin_set.append(np.linspace(self.Vp_min, self.Vp_max, N_pts))
                else:
                    c_lin_set.append(np.linspace(cmin, cmax, N_pts))

        # Create a set of matrices specifying the concentation grid for each
        # node of the network:
        C_M_SET = np.meshgrid(*c_lin_set, indexing='ij')

        M_shape = C_M_SET[0].shape

        # Create linearized arrays for each concentration, stacked into one column per node:
        c_vect_set = np.asarray([cM.ravel() for cM in C_M_SET]).T

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
                                ni:float|list=10.0,
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
            for ni in range(self.N_nodes):
                r_vect.append(ri)
        else:
            r_vect = ri

        if type(di) is not list:
            d_vect = []
            for ni in range(self.N_nodes):
                d_vect.append(di)
        else:
            d_vect = di

        self.K_vect = K_vect
        self.n_vect = n_vect
        self.r_vect = r_vect
        self.d_vect = d_vect

        # FIXME: also create the c_max vector here

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

        for ni, nt in node_data.items():
            if nt == 'Gene':
                self.node_types.append(NodeType.gene)
            elif nt == 'Process':
                self.node_types.append(NodeType.process)
            elif nt == 'Sensor':
                self.node_types.append(NodeType.sensor)
            elif nt == 'Effector':
                self.node_types.append(NodeType.effector)
            elif nt == 'Root Hub':
                self.node_types.append(NodeType.root)
            elif nt == 'Path':
                self.node_types.append(NodeType.path)
            else:
                raise Exception("Node type not found.")

    def save_network(self, filename: str):
        '''
        Write a network, including edge types, to a saved file.

        '''
        nx.write_gml(self.GG, filename)

    def read_network(self, filename: str):
        '''
        Read a network, including edge types, from a saved file.

        '''
        self.GG = nx.read_gml(filename, label=None)
        self.nodes_list = sorted(self.GG.nodes())
        self.N_nodes = len(self.nodes_list)

        self.read_edge_info_from_graph()
        self.read_node_info_from_graph()

        self.N_edges = len(self.edges_list)

        # Create numerical indices for the network:
        self._make_node_edge_indices()

        # Calculate key characteristics of the graph:
        self.characterize_graph()

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

        for ni in self.nodes_list:
            G_plt.add_node(ni,
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


    def optimized_phase_space_search(self,
                                     Ns: int=3,
                                     cmin: float=0.0,
                                     cmax: float|list=1.0,
                                     Ki: float | list = 0.5,
                                     ni: float | list = 3.0,
                                     ri: float | list = 1.0,
                                     di: float | list = 1.0,
                                     c_bounds: list|None = None,
                                     tol:float = 1.0e-15,
                                     round_sol: int=6, # decimals to round solutions to prevent duplicates
                                     method: str='Root' # Solve by finding the roots of the dc/dt equation
                                     ):
        '''

        '''

        # Create parameter vectors for the model:
        self.create_parameter_vects(Ki, ni, ri, di)

        # Determine the set of additional arguments to the optimization function:
        if self._include_process is False:
            function_args = (self.r_vect, self.d_vect, self.K_vect, self.n_vect)
        else:
            function_args = (self.r_vect, self.d_vect, self.K_vect, self.n_vect, self.process_params_f)

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
                jac_funk = self.jac_reduced_f
            else:
                N_nodes = self.N_nodes
                c_vect_s = self.c_vect_s
                dcdt_funk = self.dcdt_vect_f
                jac_funk = self.jac_f

            c_test_lin_set = []

            if type(cmax) is list: # allow for different max along each axis of the search space:
                for ci, cmi in zip(c_vect_s, cmax):
                    i = c_vect_s.index(ci)
                    if self._include_process is True and i == self._process_i:
                        c_test_lin_set.append(np.linspace(self.Vp_min, self.Vp_max, Ns))
                    else:
                        c_test_lin_set.append(np.linspace(cmin, cmi, Ns))

            else:
                for ci in c_vect_s:
                    i = c_vect_s.index(ci)
                    if self._include_process is True and i == self._process_i:
                        c_test_lin_set.append(np.linspace(self.Vp_min, self.Vp_max, Ns))
                    else:
                        c_test_lin_set.append(np.linspace(cmin, cmax, Ns))


            # Create a set of matrices specifying the concentration grid for each
            # node of the network:
            C_test_M_SET = np.meshgrid(*c_test_lin_set, indexing='ij')

            # Create linearized arrays for each concentration, stacked into one column per node:
            c_test_set = np.asarray([cM.ravel() for cM in C_test_M_SET]).T

            if c_bounds is None:
                if type(cmax) is list:
                    c_bounds = [(cmin, cmi) for i, cmi in zip(range(N_nodes), cmax)]

                else:
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
                # # FIXME: We have to be careful with this as the process might be able to assume a negative value!
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
                           fname: str=None):
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

        # Defining analytic equations for an osmotic cell volume change process:
        A_s, R_s, T_s, ni_s, m_s, V_s, Vc_s, dm_s, mu_s, Y_s, r_s = sp.symbols('A, R, T, ni, m, V, V_c, d_m, mu, Y, r',
                                                                              real=True)
        # Normalized parameters:
        Ap_s, mp_s, Ac_s, nc_s, mc_s, epsilon_s = sp.symbols('A_p, m_p, A_c, n_c, m_c, epsilon', real=True)

        dVdt_0_s = A_s ** 2 * R_s * T_s * (ni_s - m_s * V_s) / (8 * dm_s * mu_s * V_s)
        dVdt_1_s = (A_s ** 2 / (8 * dm_s * mu_s)) * (
                    R_s * T_s * ((ni_s / V_s) - m_s) - sp.Rational(4, 3) * ((Y_s * dm_s * (V_s - Vc_s) / (r_s * Vc_s))))

        # the normalized moles inside the cell is taken to be equal to an effector concentration
        # from the GRN and the normalized volume will be asigned to the c_process variable:
        # np_s = c_effector
        # Vp_s = c_process

        # Rate of change of Vp with respect to time for Vp < 1.0 is:
        dVpdt_0_s = (dVdt_0_s.subs(
            [(V_s, c_process * Vc_s), (A_s, Ap_s * Ac_s), (ni_s, nc_s * c_effector), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Rate of change of Vp with respect to time for Vp >= 1.0
        dVpdt_1_s = (dVdt_1_s.subs(
            [(V_s, c_process * Vc_s), (A_s, Ap_s * Ac_s), (ni_s, nc_s * c_effector), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Volume change rates (which are the input into the sensor node) are:
        dEdt_0_s = dVpdt_0_s
        dEdt_1_s = dVpdt_1_s

        # Piecewise function that defines this normalized-parameter osmotic cell volume change problem
        # as a strain rate:
        self.dEdt_s = sp.Piecewise((dEdt_0_s, c_process < 1.0), (dEdt_1_s, True))

        # Transform this into a numerical function:
        # self.dEdt_f = sp.lambdify([c_process, c_effector, mp_s, Ap_s, Vc_s, nc_s, mc_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s],
        #                           dVpdt_0_s)

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
            for ni in path_i:
                node_types[ni] = NodeType.path
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
                              round_sol: int=3,
                              Ki: float | list = 0.5,
                              ni: float | list = 3.0,
                              di: float | list = 1.0,
                              search_tol: float = 1.0e-15
                              ):
        '''
        By randomly generating sets of edge interaction (i.e. activator or inhibitor), find
        as many unique multistable systems as possible for a given base network.

        This does not redefine any default 'edge_types' that have been assigned to the network.

        '''
        multisols = []
        multisol_edges = []

        for i in range(N_iter):
            edge_types = self.get_edge_types(p_acti=0.5)
            self.build_analytical_model(edge_types=edge_types, add_interactions=True)
            sols_0 = self.optimized_phase_space_search(Ns=N_space,
                                                   cmax=1.0*np.max(self.in_degree_sequence),
                                                   round_sol=round_sol,
                                                   Ki = Ki,
                                                   di = di,
                                                   ni = ni,
                                                   tol=search_tol,
                                                   method="Root"
                                                  )
            if len(sols_0) >= N_multi:
                sol_char_0 = self.stability_estimate(sols_0)
                tri_char = 0
                min_val = 0.0
                for sol_dic in sol_char_0:
                    for k, v in sol_dic.items():
                        if k == 'Stability Characteristic' and v != 'Saddle Point':
                            min_val += np.sum(sol_dic['Change at Minima']**2)
                            tri_char += 1
                if tri_char >= N_multi and min_val < tol*N_multi:
                    edge_types_l = edge_types.tolist()
                    if edge_types_l not in multisol_edges: # If we don't already have this combo:
                        if verbose:
                            print(f'Found solution with {tri_char} states on iteration {i}')
                        multisols.append([sols_0, edge_types, sol_char_0])
                        multisol_edges.append(edge_types_l)

        return multisols
