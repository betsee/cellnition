#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds a probability network onto a graph model. The probability
network is based on the concept of the probability of seeing a gene product c_i.
The model is built analytically, on a fully-connected domain of nodes, where
interaction edges are +1 for activation, -1 for inhibition, and 0 for no connection.

'''
import csv
from collections.abc import Callable
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
from cellnition.science.network_enums import EdgeType, GraphType, NodeType, InterFuncType, CouplingType
from cellnition.science.gene_networks import GeneNetworkModel
import pygraphviz as pgv


class ProbabilityNet(object):
    '''
    '''

    def __init__(self,
                 gmod: GeneNetworkModel,
                 inter_func_type: InterFuncType = InterFuncType.hill,
                 coupling_type: CouplingType = CouplingType.specified):
        '''

        '''
        self._N_nodes = gmod.N_nodes
        self._gmod = gmod
        self._inter_funk_type = inter_func_type

        d_s = sp.IndexedBase('d', shape=self._gmod.N_nodes)  # Maximum rate of decay
        p_s = sp.IndexedBase('p', shape=self._gmod.N_nodes)  # Probability of gene product

        i_s, j_s = sp.symbols('i j', cls=sp.Idx)

        # Vectorized node-parameters and variables:
        self.d_vect_s = [d_s[i] for i in range(self._gmod.N_nodes)]  # maximum rate of decay for each node
        self.p_vect_s = [p_s[i] for i in range(self._gmod.N_nodes)]  # gene product probability for each node

        if self._inter_funk_type is InterFuncType.logistic:
            k_s = sp.IndexedBase('k', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Logistic coupling parameter
            mu_s = sp.IndexedBase('mu', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Logistic centre parameter
            f_inter_ij = 1 / (1 + sp.exp(-k_s[i_s, j_s] * (p_s[i_s] - mu_s[i_s, j_s])))
            self.M_k_s = sp.Array([[k_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])
            self.M_mu_s = sp.Array([[mu_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])

        else:
            beta_s = sp.IndexedBase('beta', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill centre
            n_s = sp.IndexedBase('n', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill coupling
            f_inter_ij = 1 / (1 + (p_s[i_s] * beta_s[i_s, j_s]) ** (-n_s[i_s, j_s]))
            self.M_beta_s = sp.Array([[beta_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])
            self.M_n_s = sp.Array([[n_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])

        # f_inter_term = sp.summation(f_inter_ji, (j_s, 0, self._gmod.N_nodes-1))  # evaluated sum

        # Get edge node ind pairs for the specified coupling:
        inds_edge_add, inds_edge_mult = self.set_coupling_type(coupling_type)

        f_add_list = [[] for i in range(self._gmod.N_nodes)]
        f_mult_list = [[] for i in range(self._gmod.N_nodes)]

        for (nde_i, nde_j) in inds_edge_add:
            f_add_list[nde_j].append(f_inter_ij.subs([(i_s, nde_i), (j_s, nde_j)]))

        for (nde_i, nde_j) in inds_edge_mult:
            f_mult_list[nde_j].append(f_inter_ij.subs([(i_s, nde_i), (j_s, nde_j)]))

        N_f_add = [len(fadd) for fadd in f_add_list] # normalization constants

        self.dpdt_vect_s = []
        for i in range(self._gmod.N_nodes):
            # This is the rate of change vector, roots are equilibrium points
            mult_prod = np.prod(f_mult_list[i])
            if mult_prod == 1.0:
                mult_prod = 1 # try and keep rational numbers in analytic equation, if possible

            add_prod = np.sum(f_add_list[i])
            if add_prod == 0.0:
                add_prod = 1
                N_f_add[i] = 1

            self.dpdt_vect_s.append(d_s[i]*sp.Rational(1, N_f_add[i])*add_prod*mult_prod - d_s[i]*p_s[i])

            # print(f'Add prod {np.sum(f_add_list[i])} \n Mult prod {mult_prod}')
            # print('--------')

        # # This is the rate of change vector, roots are equilibrium points
        # self.dpdt_vect_s = [d_s[i]*sp.Rational(1, self._gmod.N_nodes)*f_inter_term.subs(i_s, i) - d_s[i] * p_s[i] for i in
        #                range(self._gmod.N_nodes)]

        # This is an "energy" function to be minimized at the equilibrium points:
        self.opti_s = (sp.Matrix(self.dpdt_vect_s).T * sp.Matrix(self.dpdt_vect_s))[0, 0]


    def set_coupling_type(self, coupling_type: CouplingType):
        '''

        '''

        N_nodes = self._N_nodes

        M_n_so = sp.MatrixSymbol('M_n', N_nodes, N_nodes)

        ones_vect = sp.ones(N_nodes, 1)

        if coupling_type is CouplingType.additive:
            # Out-adjacency matrix based on conditions on the Hill exponent:
            # Suitable for pure additive or pure multiplicative interactions:
            A_add_s = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                lambda i, j: sp.Piecewise((1, M_n_so[i, j] < 0), (1, M_n_so[i, j] > 0), (0, True))
                                )

            A_mul_s = sp.ones(N_nodes, N_nodes)

        elif coupling_type is CouplingType.multiplicative:
            A_mul_s = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                lambda i, j: sp.Piecewise((1, M_n_so[i, j] < 0), (1, M_n_so[i, j] > 0), (0, True)))

            A_add_s = sp.ones(N_nodes, N_nodes)

        elif coupling_type is CouplingType.mixed:
            # Mixed interaction type:
            # Inhibitors are multiplicative; activators are additive
            A_add_s = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                lambda i, j: sp.Piecewise((1, M_n_so[i, j] > 0), (0, True)))

            A_mul_s = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                lambda i, j: sp.Piecewise((1, M_n_so[i, j] < 0), (0, True)))

        else: # else if coupling type is specified:
            pass #FIXME code this up by stepping through edge types

        n_s = sp.IndexedBase('n', shape=(N_nodes, N_nodes))

        # Create a matrix out of the n_s symbols:
        M_n_s = sp.Matrix(N_nodes, N_nodes,
                          lambda i, j: n_s[i, j])

        # Substitute it into our expression for the adjacency matrices to obtain the
        # edge count vectors essential for proper normalization of the problem:
        N_edge_vect_add = A_add_s.subs(M_n_so, sp.Matrix(M_n_s)) * ones_vect
        N_noedge_vect_add = N_nodes * sp.ones(N_nodes, 1) - N_edge_vect_add

        N_edge_vect_mul = A_mul_s.subs(M_n_so, sp.Matrix(M_n_s)) * ones_vect
        N_noedge_vect_mul = N_nodes * sp.ones(N_nodes, 1) - N_edge_vect_mul

        return N_edge_vect_add, N_edge_vect_mul, N_noedge_vect_add, N_noedge_vect_mul


        # M_inter_mult = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))
        # M_inter_add = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))
        #
        # # If 'coupling type' is 'specified' the user specifies multiplicative interactions using As and Is edges
        # if coupling_type is CouplingType.specified:
        #     for ei, ((nde_i, nde_j), edge_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
        #         if edge_type is EdgeType.A or edge_type is EdgeType.I:
        #             M_inter_add[nde_i, nde_j] = 1
        #         elif edge_type is EdgeType.Is or edge_type is EdgeType.As:
        #             M_inter_mult[nde_i, nde_j] = 1
        #
        # # 'mixed' coupling means any inhibitor is multiplicative, all activators and no-regulation is additive
        # # this is convention in standard Boolean networks:
        # elif coupling_type is CouplingType.mixed:
        #     for ei, ((nde_i, nde_j), edge_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
        #         if edge_type is EdgeType.A or edge_type is EdgeType.As:
        #             M_inter_add[nde_i, nde_j] = 1
        #         elif edge_type is EdgeType.I or edge_type is EdgeType.Is:
        #             M_inter_mult[nde_i, nde_j] = 1
        #
        # # if 'additive', all couplings are additive
        # elif coupling_type is CouplingType.additive:
        #     for ei, ((nde_i, nde_j), edge_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
        #         M_inter_add[nde_i, nde_j] = 1
        #
        # # if 'multiplicative', all couplings are multiplicative (except zero-regulated nodes are constitutive expressed)
        # elif coupling_type is CouplingType.multiplicative:
        #     for ei, ((nde_i, nde_j), edge_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
        #         M_inter_mult[nde_i, nde_j] = 1
        #
        # else:
        #     raise Exception("This CouplingType is not supported.")
        #
        # # By default, we want non-edges to be "additive" so select them this way:
        # # inds_add = (M_inter_add - M_inter_mult >= 0).nonzero()
        # # inds_mult = (M_inter_mult == 1).nonzero()
        #
        # inds_add = (M_inter_add == 1).nonzero()
        # inds_mult = (M_inter_mult == 1).nonzero()
        #
        # inds_edge_add = list(zip(inds_add[0], inds_add[1]))
        # inds_edge_mult = list(zip(inds_mult[0], inds_mult[1]))

        # return inds_edge_add, inds_edge_mult



    def make_numerical_params(self,
                       d_base: float=1.0,
                       n_base: float=3.0,
                       beta_base: float=4.0,
                       k_base: float = 10.0,
                       mu_base: float=0.25) -> None:
        '''
        Scrape the network for base parameters to initialize numerical parameters.

        '''
        # FIXME: we want to be able to input numerical lists instead of just floats
        #  for the base parameters in this input. The node parameter (d_base) will be
        #  input as a list arranged according to nodes and the edge parameters will be
        #  input as a list arranged according to the edges (and built into a matrix).

        # Scrape the network for base parameters
        self.d_vect = [d_base for i in range(self._gmod.N_nodes)]

        if self._inter_funk_type is InterFuncType.hill:
            # Scrape a network for edge types and construct a coupling matrix (Hill type functions):
            self.M_n = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))
            self.M_beta = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))

            for ei, ((nde_i, nde_j), edg_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
                self.M_beta[nde_i, nde_j] = beta_base
                if edg_type is EdgeType.A or edg_type is EdgeType.As:
                    self.M_n[nde_i, nde_j] = n_base
                else:
                    self.M_n[nde_i, nde_j] = -n_base

        elif self._inter_funk_type is InterFuncType.logistic:
            # Scrape a network for edge types and construct a coupling matrix (logistic type functions):
            self.M_k = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))
            self.M_mu = np.zeros((self._gmod.N_nodes, self._gmod.N_nodes))

            for ei, ((nde_i, nde_j), edg_type) in enumerate(zip(self._gmod.edges_index, self._gmod.edge_types)):
                self.M_mu[nde_i, nde_j] = mu_base
                if edg_type is EdgeType.A or edg_type is EdgeType.As:
                    self.M_k[nde_i, nde_j] = k_base
                else:
                    self.M_k[nde_i, nde_j] = -k_base

    def create_numerical_dpdt(self,
                              constrained_inds: list | None = None,
                              constrained_vals: list | None = None):
        '''

        '''
        # First, lambdify the change vector in a way that supports any constraints:
        if constrained_inds is None or constrained_vals is None:
            # Compute the symbolic Jacobian:
            dpdt_jac_s = sp.Matrix(self.dpdt_vect_s).jacobian(self.p_vect_s) # analytical Jacobian

            if self._inter_funk_type is InterFuncType.hill:
                dpdt_vect_f = sp.lambdify((self.p_vect_s,
                                           np.asarray(self.M_n_s),
                                           np.asarray(self.M_beta_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)

                dpdt_jac_f = sp.lambdify((self.p_vect_s,
                                          np.asarray(self.M_n_s),
                                          np.asarray(self.M_beta_s),
                                          self.d_vect_s),
                                         dpdt_jac_s)

            elif self._inter_funk_type is InterFuncType.logistic:
                dpdt_vect_f = sp.lambdify((self.p_vect_s,
                                           np.asarray(self.M_k_s),
                                           np.asarray(self.M_mu_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)

                dpdt_jac_f = sp.lambdify((self.p_vect_s,
                                          np.asarray(self.M_k_s),
                                          np.asarray(self.M_mu_s),
                                          self.d_vect_s),
                                         dpdt_jac_s)

            else:
                raise Exception("Only InterFuncType hill and logistic are supported.")


        else: # If there are constraints split the p-vals into an arguments and to-solve set:
            p_vect_args = (np.asarray(self.p_vect_s)[constrained_inds]).tolist()
            unconstrained_inds = np.setdiff1d(self._gmod.nodes_index, constrained_inds).tolist()
            p_vect_solve = (np.asarray(self.p_vect_s)[unconstrained_inds]).tolist()

            # truncate the change vector to only be for unconstrained inds:
            dpdt_vect_s = np.asarray(self.dpdt_vect_s)[unconstrained_inds].tolist()

            # Compute the symbolic Jacobian:
            dpdt_jac_s = sp.Matrix(dpdt_vect_s).jacobian(p_vect_solve) # analytical Jacobian

            if self._inter_funk_type is InterFuncType.hill:
                dpdt_vect_f = sp.lambdify((p_vect_solve,
                                           p_vect_args,
                                           np.asarray(self.M_n_s),
                                           np.asarray(self.M_beta_s),
                                           self.d_vect_s),
                                          dpdt_vect_s)

                dpdt_jac_f = sp.lambdify((p_vect_solve,
                                          p_vect_args,
                                          np.asarray(self.M_n_s),
                                          np.asarray(self.M_beta_s),
                                          self.d_vect_s),
                                         dpdt_jac_s)

            elif self._inter_funk_type is InterFuncType.logistic:
                dpdt_vect_f = sp.lambdify((p_vect_solve,
                                           p_vect_args,
                                           np.asarray(self.M_k_s),
                                           np.asarray(self.M_mu_s),
                                           self.d_vect_s),
                                          dpdt_vect_s)

                dpdt_jac_f = sp.lambdify((p_vect_solve,
                                          p_vect_args,
                                          np.asarray(self.M_k_s),
                                          np.asarray(self.M_mu_s),
                                          self.d_vect_s),
                                         dpdt_jac_s)

            else:
                raise Exception("Only InterFuncType hill and logistic are supported.")


        return dpdt_vect_f, dpdt_jac_f

    def get_function_args(self, constraint_vals: list|None=None):
        '''

        '''
        if constraint_vals is not None:
            if self._inter_funk_type is InterFuncType.hill:
                function_args = (constraint_vals, self.M_n, self.M_beta, self.d_vect)
            elif self._inter_funk_type is InterFuncType.logistic:
                function_args = (constraint_vals, self.M_k, self.M_mu, self.d_vect)
            else:
                raise Exception("Only hill and logistic InterFuncTypes are supported.")

        else:
            if self._inter_funk_type is InterFuncType.hill:
                function_args = (self.M_n, self.M_beta, self.d_vect)
            elif self._inter_funk_type is InterFuncType.logistic:
                function_args = (self.M_k, self.M_mu, self.d_vect)
            else:
                raise Exception("Only hill and logistic InterFuncTypes are supported.")

        return function_args


    def generate_state_space(self,
                             p_inds: list[int],
                             N_space: int,
                             pmin: float=1.0e-25) -> ndarray:
        '''
        Generate a discrete state space over the range of probabilities of
        each individual gene in the network.
        '''
        p_lins = []

        for i in p_inds:
            p_lins.append(np.linspace(pmin, 1.0, N_space))

        pGrid = np.meshgrid(*p_lins)

        N_pts = len(pGrid[0].ravel())

        pM = np.zeros((N_pts, self._gmod.N_nodes))

        for i, pGrid in enumerate(pGrid):
            pM[:, i] = pGrid.ravel()

        return pM

    def solve_probability_equms(self,
                                constrained_inds: list|None = None,
                                constrained_vals: list|None = None,
                                d_base: float = 1.0,
                                n_base: float = 3.0,
                                beta_base: float = 4.0,
                                k_base: float = 10.0,
                                mu_base: float = 0.25,
                                N_space: int = 2,
                                pmin: float=1.0e-9,
                                tol: float=1.0e-15,
                                N_round_sol: int = 2,
                                jac_derivatives_cols: bool=False,
                                ):
        '''
        Solve for the equilibrium points of gene product probabilities in
        terms of a given set of numerical parameters.
        '''

        dpdt_vect_f, dpdt_jac_f = self.create_numerical_dpdt(constrained_inds=constrained_inds,
                                                 constrained_vals=constrained_vals)

        self.make_numerical_params(d_base=d_base,
                                   n_base=n_base,
                                   beta_base=beta_base,
                                   k_base=k_base,
                                   mu_base=mu_base)

        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self._gmod.nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self._gmod.nodes_index, constrained_inds).tolist()

        M_pstates = self.generate_state_space(unconstrained_inds, N_space, pmin)

        sol_Mo = []

        function_args = self.get_function_args(constraint_vals=constrained_vals)

        for pvecto in M_pstates: # for each test vector:
            p_vect_sol = pvecto[unconstrained_inds] # get values for the genes we're solving for...

            sol_roots = fsolve(dpdt_vect_f,
                               p_vect_sol,
                               args=function_args,
                               xtol=tol,
                               fprime=dpdt_jac_f,
                               col_deriv=jac_derivatives_cols
                               )

            p_eqms = np.zeros(self._gmod.N_nodes)
            p_eqms[unconstrained_inds] = sol_roots

            if constrained_inds is not None and constrained_vals is not None:
                p_eqms[constrained_inds] = constrained_vals

            sol_Mo.append(p_eqms)

        _, unique_inds = np.unique(np.round(sol_Mo, N_round_sol), axis=0, return_index=True)

        sol_M = np.asarray(sol_Mo)[unique_inds]

        stable_sol_M, sol_M_char = self.find_attractor_sols(sol_M,
                                                             dpdt_vect_f,
                                                             dpdt_jac_f,
                                                             function_args,
                                                             constrained_inds=constrained_inds,
                                                             tol= 1.0e-1,
                                                             verbose = True,
                                                             unique_sols = True,
                                                             sol_round = 1,
                                                             save_file = None)

        return stable_sol_M, sol_M_char, sol_M

    def find_attractor_sols(self,
                            sols_0: list|ndarray,
                            dpdt_vect_f: Callable,
                            jac_f: Callable,
                            func_args: tuple|list,
                            constrained_inds: list | None = None,
                            tol: float=1.0e-1,
                            verbose: bool=True,
                            unique_sols: bool = True,
                            sol_round: int = 1,
                            save_file: str|None = None
                            ):
        '''

        '''

        eps = 1.0e-25  # we need a small value to add to avoid dividing by zero

        sol_dicts_list = []

        if constrained_inds is None:
            unconstrained_inds = self._gmod.nodes_index

        else:
            unconstrained_inds = np.setdiff1d(self._gmod.nodes_index, constrained_inds)

        for pminso in sols_0:

            solution_dict = {}

            solution_dict['Minima Values'] = pminso

            pmins = np.asarray(pminso) # add the small amount here, before calculating the jacobian

            solution_dict['Change at Minima'] = dpdt_vect_f(pmins[unconstrained_inds], *func_args)

            jac = jac_f(pmins[unconstrained_inds], *func_args)

            # get the eigenvalues of the jacobian at this equillibrium point:
            eig_valso, eig_vects = np.linalg.eig(jac)

            # round the eigenvalues so we don't have issue with small imaginary components
            eig_vals = np.round(np.real(eig_valso), 1) + np.round(np.imag(eig_valso), 1) * 1j

            solution_dict['Jacobian Eigenvalues'] = eig_vals

            # get the indices of eigenvalues that have only real components:
            real_eig_inds = (np.imag(eig_vals) == 0.0).nonzero()[0]

            # FIXME: this should be enumeration
            # If all eigenvalues are real and they're all negative:
            if len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) <= 0.0):
                char_tag = 'Stable Attractor'

            # If all eigenvalues are real and they're all positive:
            elif len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) > 0.0):
                char_tag = 'Stable Repellor'

            # If there are no real eigenvalues we only know its a limit cycle but can't say
            # anything certain about stability:
            elif len(real_eig_inds) == 0 and np.all(np.real(eig_vals) <= 0.0):
                char_tag = 'Stable Limit Cycle'

            # If there are no real eigenvalues and a mix of real component sign, we only know its a limit cycle but can't say
            # anything certain about stability:
            elif len(real_eig_inds) == 0 and np.any(np.real(eig_vals) > 0.0):
                char_tag = 'Limit Cycle'

            elif np.all(np.real(eig_vals[real_eig_inds]) <= 0.0):
                char_tag = 'Stable Limit Cycle'

            elif np.any(np.real(eig_vals[real_eig_inds] > 0.0)):
                char_tag = 'Saddle Point'
            else:
                char_tag = 'Undetermined'

            solution_dict['Stability Characteristic'] = char_tag

            sol_dicts_list.append(solution_dict)

        solsM = []
        sol_char_list = []
        sol_char_error = []
        i = 0
        for sol_dic in sol_dicts_list:
            error = np.sum(np.asarray(sol_dic['Change at Minima'])**2)
            char = sol_dic['Stability Characteristic']
            sols = sol_dic['Minima Values']

            if char != 'Saddle Point' and error <= tol:
                i += 1
                if verbose and unique_sols is False:
                    print(f'Soln {i}, {char}, {sols}, {np.round(error, sol_round)}')
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

        # if save_file is not None:
        #     solsMi = np.asarray(solsM)
        #     header = [f'State {i}' for i in range(solsMi.shape[0])]
        #     with open(save_file, 'w', newline="") as file:
        #         csvwriter = csv.writer(file)  # create a csvwriter object
        #         csvwriter.writerow(header)  # write the header
        #         csvwriter.writerow(sol_char_error)  # write the root error at steady-state
        #         csvwriter.writerow(sol_char_list)  # write the attractor characterization
        #         for si in solsMi.T:
        #             csvwriter.writerow(si)  # write the soln data rows for each gene

        return solsM_return, sol_dicts_list


