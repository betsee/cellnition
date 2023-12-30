#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module
'''

import numpy as np
from numpy import ndarray
import networkx as nx
import sympy as sp
from cellnition.science.enumerations import EdgeType
import pyvista as pv

class GeneNetworkModel(object):
    '''

    '''

    def __init__(self,
                 N_nodes: int,
                 edges: list|ndarray|None = None,
                 beta: float = 0.20,
                 gamma: float=0.75,
                 delta: float=0.0):
        '''

        '''
        self.N_nodes = N_nodes # number of nodes in the network

        # Depending on whether edges are supplied by user, generate
        # a graph:

        if edges is None:
            self.generate_scale_free_network(beta,
                                             gamma,
                                             delta_in=delta,
                                             delta_out=delta)

        else:
            self.edges_list = edges
            self.GG = nx.DiGraph(self.edges_list)
            self.N_edges = len(self.edges_list)
            self.nodes_list = sorted(self.GG.nodes())

        # Calculate key characteristics of the graph
        self._characterize_graph()



    def generate_scale_free_network(self,
                         beta: float,
                         gamma: float,
                         delta_in: float,
                         delta_out: float):
        '''

        '''

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

        # obtain the unique edges only:
        self.edges_list = list(set(GGo.edges()))
        self.N_edges = len(self.edges_list)

        # As the scale_free_graph function can return duplicate edges, get rid of these
        # by re-defining the graph with the unique edges only:
        GG = nx.DiGraph(self.edges_list)
        self.nodes_list = sorted(GG.nodes())
        self.GG = GG

    def _characterize_graph(self):
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

    def get_edge_types(self, p_acti: float=0.5):
        '''
        Automatically generate a conse
        rved edge-type vector for use in
        model building.
        '''

        p_inhi = 1.0 - p_acti

        edge_types_o = [EdgeType.A, EdgeType.I]
        edge_prob = [p_acti, p_inhi]
        edge_types = np.random.choice(edge_types_o, self.N_edges, edge_prob)

        return edge_types

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

        self.edge_funcs = []
        for et in self.edge_types:
            if et is EdgeType.A:
                self.edge_funcs.append(self.f_acti_s)
            else:
                self.edge_funcs.append(self.f_inhi_s)


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
        self.dcdt_vect_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                        self.d_vect_s, self.K_vect_s, self.n_vect_s], self.dcdt_vect_s)

        self.jac_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                  self.d_vect_s, self.K_vect_s, self.n_vect_s], self.jac_s)

        self.hess_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                  self.d_vect_s, self.K_vect_s, self.n_vect_s], self.hess_s)

        self.opti_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                  self.d_vect_s, self.K_vect_s, self.n_vect_s], self.opti_s)

        self.opti_jac_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                  self.d_vect_s, self.K_vect_s, self.n_vect_s], self.opti_jac_s)

        self.opti_hess_f = sp.lambdify([self.c_vect_s, self.r_vect_s,
                                  self.d_vect_s, self.K_vect_s, self.n_vect_s], self.opti_hess_s)


    def f_acti_s(self, cc, kk, nn):
        '''

        '''
        return ((cc / kk) ** nn) / (1 + (cc / kk) ** nn)

    def f_inhi_s(self, cc, kk, nn):
        '''

        '''
        return 1 / (1 + (cc / kk) ** nn)


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

        pl = pv.Plotter()
        pl.add_mesh(streamlines.tube(radius=tube_radius), lighting=lighting, cmap=cmap)
        pl.remove_scalar_bar("vectors")
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
                                additive_interactions: bool=False):
        '''

        '''

        # Build an analytical model based on the edge types and other supplied info:
        self.build_analytical_model(prob_acti=prob_acti, edge_types=edge_types, add_interactions=additive_interactions)

        # Create linear set of concentrations over the desired range
        # for each node of the network:
        c_lin_set = []
        for i in range(self.N_nodes):
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
            dcdt_i = self.dcdt_vect_f(c_vecti, r_vect, d_vect, K_vect, n_vect).flatten()
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