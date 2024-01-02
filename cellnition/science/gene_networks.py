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
from scipy.optimize import minimize
import networkx as nx
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.enumerations import EdgeType, GraphType, NodeType
from cellnition.science.stability import Solution
import pygraphviz as pgv
import pyvista as pv

# TODO: Color the master hub(s) and/or a leaf and/or sensor and/or process nodes in different colours
# TODO: Allow process to be added to the network (node with different physics)
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
            self.generate_special_network(beta=beta,
                                          gamma=gamma,
                                          graph_type=graph_type,
                                          delta_in=delta,
                                          delta_out=delta,
                                          p_edge=p_edge)

        else:
            self.edges_list = edges
            self.GG = nx.DiGraph(self.edges_list)
            self.N_edges = len(self.edges_list)
            # self.nodes_list = sorted(self.GG.nodes())
            self.nodes_list = np.arange(self.N_nodes)

        # Calculate key characteristics of the graph
        self.characterize_graph()



    def generate_special_network(self,
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
        # self.nodes_list = sorted(GG.nodes())
        self.nodes_list = np.arange(self.N_nodes)
        self.GG = GG

    def characterize_graph(self):
        '''

        '''
        # Degree analysis:
        self.in_degree_sequence = np.asarray(sorted((d for n, d in self.GG.in_degree()), reverse=True))
        self.in_dmax = self.in_degree_sequence.max()

        self.out_degree_sequence = np.asarray(sorted((d for n, d in self.GG.out_degree()), reverse=True))
        self.out_dmax = self.out_degree_sequence.max()

        self.in_bins, self.in_degree_counts = np.unique(self.in_degree_sequence, return_counts=True)
        self.out_bins, self.out_degree_counts = np.unique(self.out_degree_sequence, return_counts=True)

        # Nodes sorted by number of out-degree edges:
        self.nodes_by_out_degree = [ni for ni, di
                                    in sorted(self.GG.out_degree, key=lambda x: x[1], reverse=True)]

        self.nodes_by_in_degree = [ni for ni, di
                                    in sorted(self.GG.in_degree, key=lambda x: x[1], reverse=True)]


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
        for mn_i in range(len(self.nodes_list)):
            number_paths_to_i = []
            for i in range(len(self.nodes_list)):
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
        for ni, nt in zip(self.nodes_list, node_types):
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

        if edge_types is None:
            self.edge_types = self.get_edge_types(p_acti=prob_acti)

        else:
            self.edge_types = edge_types

        self.set_edge_types(self.edge_types)

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_list]  # Set all nodes to the gene type

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
        self.r_vect_s = [r_max_s[i] for i in self.nodes_list]
        self.d_vect_s = [d_max_s[i] for i in self.nodes_list]
        self.c_vect_s = [c_s[i] for i in self.nodes_list]

        efunc_vect = [[] for i in self.nodes_list]
        for ei, ((i, j), fun_type) in enumerate(zip(self.edges_list, self.edge_funcs)):
            efunc_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))

        dcdt_vect_s = []

        for ni, fval_set in enumerate(efunc_vect):
            if add_interactions:
                if len(fval_set) == 0:
                    normf = 1
                else:
                    normf = sp.Rational(1, len(fval_set))

                dcdt_vect_s.append(r_max_s[ni] * np.sum(fval_set)*normf - c_s[ni] * d_max_s[ni])
            else:
                dcdt_vect_s.append(r_max_s[ni] * np.prod(fval_set) - c_s[ni] * d_max_s[ni])

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
            self.nodes_list = np.arange(self.N_nodes)  # We make a new nodes list

            # Harvest data from edge attributes for edge property re-assignment:
            self.read_edge_info_from_graph()

            # Indices of key new edges:
            self.ei_process_sensor = self.edges_list.index((self._process_i, self._sensor_i))
            self.ei_sensor_root = self.edges_list.index((self._sensor_i, self._root_i))

            # Re-calculate key characteristics of the graph after adding in new nodes and edges:
            self.characterize_graph()

        else:
            self._root_i = control_node_dict['root']
            self._effector_i = control_node_dict['effector']
            self._sensor_i = control_node_dict['sensor']
            self._process_i = control_node_dict['process']

            # Indices of key new edges:
            self.ei_process_sensor = self.edges_list.index((self._process_i, self._sensor_i))
            self.ei_effector_process = self.edges_list.index((self._effector_i, self._process_i))

            # Override the edge-type for the control loop effector-process:
            self.edge_types[self.ei_effector_process] = EdgeType.N # This is always neutral

            # Update the edge types on the graph edges:
            self.set_edge_types(self.edge_types)

        # Now that indices are set, give nodes a type attribute:
        node_types = [NodeType.gene for i in self.nodes_list]  # Set all nodes to the gene type
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
        self.r_vect_s = [r_max_s[i] for i in self.nodes_list]
        self.d_vect_s = [d_max_s[i] for i in self.nodes_list]
        self.c_vect_s = [c_s[i] for i in self.nodes_list]

        # Create the analytic equations governing the process:
        self.set_analytic_process(c_s[self._sensor_i], c_s[self._process_i])

        # Create the edge-function collections at each node for the GRN interactions:
        efunc_vect = [[] for i in self.nodes_list]
        for ei, ((i, j), fun_type) in enumerate(zip(self.edges_list, self.edge_funcs)):
            efunc_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))

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

                    dcdt_vect_s.append(r_max_s[ni] * np.sum(fval_set) * normf - c_s[ni] * d_max_s[ni])
                else:
                    dcdt_vect_s.append(r_max_s[ni] * np.prod(fval_set) - c_s[ni] * d_max_s[ni])

        # analytical rate of change of concentration vector for the network:
        self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)

        self._include_process = True # Set the internal boolean to True for consistency
        # Generate the optimization "energy" function as well as jacobians and hessians for the system:
        self._generate_optimization_functions()


    def _generate_optimization_functions(self):
        '''

        '''

        if self._include_process:
            lambda_params = [self.c_vect_s,
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
                             self.n_vect_s]

        # Create a Jacobian for the system
        self.jac_s = self.dcdt_vect_s.jacobian(sp.Matrix(self.c_vect_s)).applyfunc(sp.simplify)

        # The Hessian is a more complex tensor:
        self.hess_s = sp.Array(
            [[[self.dcdt_vect_s[i].diff(dcj).diff(dci) for dcj in self.c_vect_s]
              for dci in self.c_vect_s] for i in range(self.N_nodes)])

        # Optimization function for solving the problem:
        self.opti_s = (self.dcdt_vect_s.T*self.dcdt_vect_s)[0]

        self.opti_jac_s = sp.Array([self.opti_s.diff(ci) for ci in self.c_vect_s])

        self.opti_hess_s = sp.Matrix(self.opti_jac_s).jacobian(self.c_vect_s)

        # Lambdify the two outputs so they can be used to study the network numerically:
        self.dcdt_vect_f = sp.lambdify(lambda_params, self.dcdt_vect_s)

        self.jac_f = sp.lambdify(lambda_params, self.jac_s)

        self.hess_f = sp.lambdify(lambda_params, self.hess_s)

        self.opti_f = sp.lambdify(lambda_params, self.opti_s)

        self.opti_jac_f = sp.lambdify(lambda_params, self.opti_jac_s)

        self.opti_hess_f = sp.lambdify(lambda_params, self.opti_hess_s)


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
                                edge_types: list|ndarray|None=None,
                                Nc: int=15,
                                cmin: float=0.0,
                                cmax: float=1.0,
                                Ki: float|list=0.5,
                                ni:float|list=10.0,
                                ri:float|list=1.0,
                                di:float|list=1.0,
                                zer_thresh: float=0.01,
                                prob_acti: float=0.5,
                                additive_interactions: bool=False,
                                include_process: bool = False
                                ):
        '''

        '''

        if include_process is False:
            # Build an analytical model based on the edge types and other supplied info:
            self.build_analytical_model(prob_acti=prob_acti,
                                        edge_types=edge_types,
                                        add_interactions=additive_interactions)

        else:
            # Build an analytical model with the osmotic process included in the
            # heterogeneous network model:
            self.build_analytical_model_with_process(prob_acti=prob_acti,
                                                     edge_types=edge_types,
                                                     add_interactions=additive_interactions)

        # Create linear set of concentrations over the desired range
        # for each node of the network:
        c_lin_set = []
        for i in range(self.N_nodes):
            if self._include_process and i == self._process_i:
                c_lin_set.append(np.linspace(self.epsilon_min, self.epsilon_max, Nc))
            else:
                c_lin_set.append(np.linspace(cmin, cmax, Nc))



        # Create a set of matrices specifying the concentation grid for each
        # node of the network:
        C_M_SET = np.meshgrid(*c_lin_set, indexing='ij')

        M_shape = C_M_SET[0].shape

        # Create linearized arrays for each concentration, stacked into one column per node:
        c_vect_set = np.asarray([cM.ravel() for cM in C_M_SET]).T

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

        dcdt_M = np.zeros(c_vect_set.shape)

        for i, c_vecti in enumerate(c_vect_set):
            if self._include_process is False:
                dcdt_i = self.dcdt_vect_f(c_vecti, r_vect, d_vect, K_vect, n_vect).flatten()
            else:
                dcdt_i = self.dcdt_vect_f(c_vecti,
                                          r_vect,
                                          d_vect,
                                          K_vect,
                                          n_vect,
                                          self.process_params_f).flatten()
            dcdt_M[i] = dcdt_i * 1

        dcdt_M_set = []
        for dci in dcdt_M.T:
            dcdt_M_set.append(dci.reshape(M_shape))

        self.c_lin_set = c_lin_set
        self.C_M_SET = C_M_SET
        self.M_shape = M_shape

        self.K_vect = K_vect
        self.n_vect = n_vect
        self.r_vect = r_vect
        self.d_vect = d_vect

        self.dcdt_M_set = np.asarray(dcdt_M_set)
        self.dcdt_dmag = np.sqrt(np.sum(self.dcdt_M_set ** 2, axis=0))
        self.dcdt_zeros = ((self.dcdt_dmag / self.dcdt_dmag.max()) < zer_thresh).nonzero()

        return self.dcdt_zeros, self.dcdt_M_set, self.dcdt_dmag, self.c_lin_set, self.C_M_SET

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
            else:
                raise Exception("Node type not found.")

    def save_network(self, filename: str):
        '''
        Write a network, including edge types, from a saved file.

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
                                     Ns: int=2,
                                     cmin: float=0.0,
                                     cmax: float=1.0,
                                     Ki: float | list = 0.5,
                                     ni: float | list = 10.0,
                                     ri: float | list = 1.0,
                                     di: float | list = 1.0,
                                     c_bounds: list|None = None,
                                     zer_thresh: float=0.001,
                                     fast_solver: bool=False,
                                     ):
        '''

        '''

        c_test_lin_set = []
        for i in range(self.N_nodes):
            if self._include_process is True and i == self._process_i:
                c_test_lin_set.append(np.linspace(self.epsilon_min, self.epsilon_max, Ns))
            else:
                c_test_lin_set.append(np.linspace(cmin, cmax, Ns))


        # Create a set of matrices specifying the concentation grid for each
        # node of the network:
        C_test_M_SET = np.meshgrid(*c_test_lin_set, indexing='ij')

        # Create linearized arrays for each concentration, stacked into one column per node:
        c_test_set = np.asarray([cM.ravel() for cM in C_test_M_SET]).T

        # print(f"Estimated compute time: {(c_test_set.shape[0]*0.0123)/60} minutes")

        if type(Ki) != list:
            K_vect = (Ki * np.ones(self.N_edges)).tolist()
        else:
            K_vect = Ki

        if type(ni) != list:
            n_vect = (ni * np.ones(self.N_edges)).tolist()
        else:
            n_vect = ni

        if type(ri) != list:
            r_vect = (ri*np.ones(self.N_nodes)).tolist()
        else:
            r_vect = ri

        if type(di) != list:
            d_vect = (di*np.ones(self.N_nodes)).tolist()
        else:
            d_vect = di

        if c_bounds is None:
            c_bounds = [(cmin, cmax) for i in range(self.N_nodes)]

        if self._include_process is False:
            function_args = (r_vect, d_vect, K_vect, n_vect)
        else:
            function_args = (r_vect, d_vect, K_vect, n_vect, self.process_params_f)
            c_bounds[self._process_i] = (self.epsilon_min, self.epsilon_max)

        mins_found = set()

        for c_vecti in c_test_set:

            if fast_solver is False:
                sol0 = minimize(self.opti_f,
                                c_vecti,
                                args=function_args,
                                method='Powell',
                                hessp=None,
                                bounds=c_bounds,
                                tol=None,
                                callback=None,
                                options=None)

                if sol0.fun < zer_thresh:
                    mins_found.add(tuple(np.round(sol0.x, 1)))

            else: # just check to see if the conc vector is a solution
                if self._include_process is False:
                    fun_val = self.opti_f(c_vecti, r_vect, d_vect, K_vect, n_vect)
                else:
                    fun_val = self.opti_f(c_vecti, r_vect, d_vect, K_vect, n_vect, self.process_params_f)

                if fun_val < zer_thresh:
                    mins_found.add(tuple(c_vecti))

        self.mins_found = mins_found
        self.r_vect = r_vect
        self.d_vect = d_vect
        self.K_vect = K_vect
        self.n_vect = n_vect

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

            # print(f'dcdt at min: {self.dcdt_vect_f(cmins, r_vect, d_vect, K_vect, n_vect).flatten()}')

            solution_dict['Change at Minima'] = self.dcdt_vect_f(*func_args).flatten()

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


    def set_analytic_process(self, c_sensor: Symbol|Indexed, c_process: Symbol|Indexed):
        '''
        cs : Symbol
            Symbolic concentration from a node in the GRN network that represents the moles of
            osmolyte inside the cell.

        '''

        # Defining analytic equations for an osmotic cell volume change process:
        A_s, R_s, T_s, n_s, m_s, V_s, Vc_s, dm_s, mu_s, Y_s, r_s = sp.symbols('A, R, T, n, m, V, V_c, d_m, mu, Y, r',
                                                                              real=True)
        # Normalized parameters:
        Ap_s, mp_s, Ac_s, nc_s, mc_s, epsilon_s = sp.symbols('A_p, m_p, A_c, n_c, m_c, epsilon', real=True)

        dVdt_0_s = A_s ** 2 * R_s * T_s * (n_s - m_s * V_s) / (8 * dm_s * mu_s * V_s)
        dVdt_1_s = (A_s ** 2 / (8 * dm_s * mu_s)) * (
                    R_s * T_s * ((n_s / V_s) - m_s) - sp.Rational(4, 3) * ((Y_s * dm_s * (V_s - Vc_s) / (r_s * Vc_s))))

        # the normalized moles inside the cell is taken to be equal to an effector concentration
        # from the GRN and the normalized volume will be asigned to the c_process variable:
        np_s = c_sensor
        Vp_s = c_process

        # Rate of change of Vp with respect to time for Vp < 1.0 is:
        dVpdt_0_s = (dVdt_0_s.subs(
            [(V_s, Vp_s * Vc_s), (A_s, Ap_s * Ac_s), (n_s, nc_s * np_s), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Rate of change of Vp with respect to time for Vp >= 1.0
        dVpdt_1_s = (dVdt_1_s.subs(
            [(V_s, Vp_s * Vc_s), (A_s, Ap_s * Ac_s), (n_s, nc_s * np_s), (m_s, mc_s * mp_s)]) / Vc_s).simplify()

        # Strain rates (which are the input into the sensor node) are:
        dEdt_0_s = dVpdt_0_s - 1
        dEdt_1_s = dVpdt_1_s - 1

        # Piecewise function that defines this normalized-parameter osmotic cell volume change problem
        # as a strain rate:
        self.dEdt_s = sp.Piecewise((dEdt_0_s, Vp_s < 1.0), (dEdt_1_s, True))

        # Transform this into a numerical function:
        self.dEdt_f = sp.lambdify([Vp_s, np_s, mp_s, Ap_s, Vc_s, nc_s, mc_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s],
                                  dVpdt_0_s)

        # Go ahead and initialize some parameters for this process function: # FIXME these need to be
        # made easier to input, vary and change:
        self.m_f = 0.5  # Normalized environmental osmolyte concentration
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

        self.epsilon_min = -0.8 # minimum strain that can be achieved
        self.epsilon_max = 2.0 # maximum strain that can be achieved

        # symbolic parameters for the dV/dt process (these must be augmented onto the GRN parameters
        # when lambdifying):
        self.process_params_s = (mp_s, Ap_s, Vc_s, nc_s, mc_s, Ac_s, R_s, T_s, Y_s, dm_s, mu_s, r_s)

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

        # evaluate the function as:
        # self.dEdt_f(self.V_f,
        #             self.n_f,
        #             self.m_f,
        #             self.A_f,
        #             self.Vc_f,
        #             self.nc_f,
        #             self.mc_f,
        #             self.Ac_f,
        #             self.R_f,
        #             self.T_f,
        #             self.Y_f,
        #             self.dm_f,
        #             self.mu_f,
        #             self.r_f)
