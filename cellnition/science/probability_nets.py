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
from sympy import MutableDenseMatrix
from cellnition.science.network_enums import EdgeType, GraphType, NodeType, InterFuncType, CouplingType
from cellnition.science.gene_networks import GeneNetworkModel
import pygraphviz as pgv

# FIXME: Do this so we can use user-specified edge types in the network construction
# FIXME: Do this so we can build a network from the M_n matrix

class ProbabilityNet(object):
    '''
    '''
    def __init__(self,
                 gmod: GeneNetworkModel,
                 coupling_type: CouplingType = CouplingType.specified):
        '''

        '''
        self._N_nodes = gmod.N_nodes
        self._gmod = gmod

        d_s = sp.IndexedBase('d', shape=self._N_nodes)  # Maximum rate of decay
        p_s = sp.IndexedBase('p', shape=self._N_nodes)  # Probability of gene product

        # Vectorized node-parameters and variables:
        self.d_vect_s = sp.Matrix([d_s[i] for i in range(self._N_nodes)])  # maximum rate of decay for each node
        self.p_vect_s = sp.Matrix([p_s[i] for i in range(self._N_nodes)])  # gene product probability for each node

        beta_s = sp.IndexedBase('beta', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill centre
        n_s = sp.IndexedBase('n', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill coupling

        # Create a matrix out of the n_s symbols:
        self.M_n_s = sp.Matrix(self._N_nodes, self._N_nodes,
                          lambda i, j: n_s[i, j])


        self.M_beta_s = sp.Matrix(self._N_nodes, self._N_nodes,
                             lambda i, j: beta_s[i, j])

        # Define vector of ones to use in matrix operations:
        ones_vect = sp.ones(1, self._N_nodes)

        # Create a matrix that allows us to access the concentration vectors
        # duplicated along columns:
        M_p_s = self.p_vect_s * ones_vect

        # Matrix symbols to construct matrix equation bases:
        M_n_so = sp.MatrixSymbol('M_n', self._N_nodes, self._N_nodes)
        M_beta_so = sp.MatrixSymbol('M_beta', self._N_nodes, self._N_nodes)
        M_p_so = sp.MatrixSymbol('M_p', self._N_nodes, self._N_nodes)

        if coupling_type is CouplingType.additive:
            # If coupling type is pure additive (cooperation/competition
            # between all activators and inhibitors):
            M_funk_add_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                      lambda i, j: sp.Piecewise(((1 / (
                                                  1 + (M_beta_so[j, i] * M_p_so[j, i]) ** (M_n_so[j, i]))),
                                                                 M_n_so[j, i] < 0),
                                                                ((1 / (1 + (M_beta_so[j, i] * M_p_so[j, i]) ** (
                                                                M_n_so[j, i]))), M_n_so[j, i] > 0),
                                                                (0, True)))

            M_funk_mul_so = sp.ones(self._N_nodes, self._N_nodes)

            # prepare an adjacency matrix for the addition portion
            A_add_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                 lambda i, j: sp.Piecewise((1, M_n_so[j, i] < 0),
                                                           (1, M_n_so[j, i] > 0), (0, True))
                                 )

        elif coupling_type is CouplingType.multiplicative:
            # If coupling type is pure multiplicative:
            M_funk_mul_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                      lambda i, j: sp.Piecewise(((1 / (
                                                  1 + (M_beta_so[j, i] * M_p_so[j, i]) ** (M_n_so[j, i]))),
                                                                 M_n_so[j, i] < 0),
                                                                ((1/(1 + (M_beta_so[j, i]*M_p_so[j, i])**(
                                                                M_n_so[j, i]))), M_n_so[j, i] > 0),
                                                                (1, True)))

            # When we know all additive interactions are None, prepare a split-prob matrix:
            M_funk_add_so = sp.ones(self._N_nodes, self._N_nodes)

            A_add_so = sp.ones(self._N_nodes, self._N_nodes)

        elif coupling_type is CouplingType.mixed:
            # If coupling type is mixed where inhibitors are dominant and activators cooperate:
            M_funk_add_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                      lambda i, j: sp.Piecewise(
                                          ((1 / (1 + (M_beta_so[j, i] * M_p_so[j, i]) ** (M_n_so[j, i]))),
                                           M_n_so[j, i] > 0),
                                          (0, True)))

            M_funk_mul_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                      lambda i, j: sp.Piecewise(((1 / (
                                                  1 + (M_beta_so[j, i] * M_p_so[j, i]) ** (M_n_so[j, i]))),
                                                                 M_n_so[j, i] < 0),
                                                                (1, True)))

            A_add_so = sp.Matrix(M_n_so.rows, M_n_so.cols,
                                 lambda i, j: sp.Piecewise((1, M_n_so[j, i] > 0), (0, True)))

        else: # FIXME: add in specified
            raise Exception("Only additive, multiplicative, and mixed couplings supported.")

        M_funk_add_s = M_funk_add_so.subs([(M_p_so, M_p_s),
                                           (M_beta_so, self.M_beta_s),
                                           (M_n_so, self.M_n_s)
                                           ]
                                          )

        M_funk_mul_s = M_funk_mul_so.subs([(M_p_so, M_p_s),
                                           (M_beta_so, self.M_beta_s),
                                           (M_n_so, self.M_n_s)])

        A_add_s = A_add_so.subs(M_n_so, self.M_n_s)

        self._n_add_edges = A_add_s * ones_vect.T
        self._add_terms = M_funk_add_s * ones_vect.T
        self._mul_terms = sp.Matrix(np.product(M_funk_mul_s, axis=1))

    def construct_dpdt_vect(self, nn: ndarray):
        '''

        '''
        # substitute in real values for the Hill exponent, in order
        # to solve the piecewise conditions:
        subs_list = self.subs_matrix_vals(self.M_n_s, nn)
        n_add_edges = self._n_add_edges.subs(subs_list)
        add_terms = self._add_terms.subs(subs_list)
        mul_terms = self._mul_terms.subs(subs_list)

        self.dpdt_vect_s = []

        for i in range(self._N_nodes):
            # Fix a potential 0/0 issue:
            if n_add_edges[i] == 0.0 and add_terms[i] == 0.0:
                print('Divide by zero found')
                n_add_edges[i] = 1

            print(f'{i}: n_add_edges: {n_add_edges[i]} \n '
                  f'add_term: {add_terms[i]} \n '
                  f'mul_term: {mul_terms[i]}')

            print('------')

            self.dpdt_vect_s.append(self.d_vect_s[i]*mul_terms[i]*(1/n_add_edges[i])*add_terms[i] -
                            self.p_vect_s[i]*self.d_vect_s[i])

        # This is an "energy" function to be minimized at the equilibrium points:
        self.opti_s = (sp.Matrix(self.dpdt_vect_s).T * sp.Matrix(self.dpdt_vect_s))[0, 0]


    def subs_matrix_vals(self, M_s: MutableDenseMatrix, M_vals: MutableDenseMatrix|ndarray):
        '''

        '''
        subs_list = []
        for i in range(self._N_nodes):
            for j in range(self._N_nodes):
                subs_list.append((M_s[i, j], M_vals[i, j]))

        return subs_list

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


