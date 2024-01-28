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
from cellnition.science.interaction_functions import (f_acti_hill_s,
                                                      f_inhi_hill_s,
                                                      f_neut_s,
                                                      f_acti_logi_s,
                                                      f_inhi_logi_s)
from cellnition.science.gene_networks import GeneNetworkModel
import pygraphviz as pgv


class ProbabilityNet(object):
    '''
    '''

    def __init__(self,
                 gmod: GeneNetworkModel,
                 inter_func_type: InterFuncType = InterFuncType.hill):
        '''

        '''
        self._gmod = gmod
        self._inter_funk_type = inter_func_type

        # FIXME: Need to also consider coupling type

        d_s = sp.IndexedBase('d', shape=self._gmod.N_nodes)  # Maximum rate of decay
        p_s = sp.IndexedBase('p', shape=self._gmod.N_nodes)  # Probability of gene product

        i_s, j_s = sp.symbols('i j', cls=sp.Idx)

        # Vectorized node-parameters and variables:
        self.d_vect_s = [d_s[i] for i in range(self._gmod.N_nodes)]  # maximum rate of decay for each node
        self.p_vect_s = [p_s[i] for i in range(self._gmod.N_nodes)]  # gene product probability for each node

        if self._inter_funk_type is InterFuncType.logistic:
            k_s = sp.IndexedBase('k', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Logistic coupling parameter
            mu_s = sp.IndexedBase('mu', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Logistic centre parameter
            f_inter_ji = 1 / (1 + sp.exp(-k_s[j_s, i_s] * (p_s[j_s] - mu_s[j_s, i_s])))
            self.M_k_s = sp.Array([[k_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])
            self.M_mu_s = sp.Array([[mu_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])

        else:
            beta_s = sp.IndexedBase('beta', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill centre
            n_s = sp.IndexedBase('n', shape=(self._gmod.N_nodes, self._gmod.N_nodes))  # Hill coupling
            f_inter_ji = 1 / (1 + (p_s[j_s] * beta_s[j_s, i_s]) ** (-n_s[j_s, i_s]))
            self.M_beta_s = sp.Array([[beta_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])
            self.M_n_s = sp.Array([[n_s[i, j] for j in range(self._gmod.N_nodes)] for i in range(self._gmod.N_nodes)])

        f_inter_term = sp.summation(f_inter_ji, (j_s, 0, self._gmod.N_nodes-1))  # evaluated sum

        # This is the rate of change vector, roots are equilibrium points
        self.dpdt_vect_s = [d_s[i]*sp.Rational(1, self._gmod.N_nodes)*f_inter_term.subs(i_s, i) - d_s[i] * p_s[i] for i in
                       range(self._gmod.N_nodes)]

        # This is an "energy" function to be minimized at the equilibrium points:
        self.opti_s = (sp.Matrix(self.dpdt_vect_s).T * sp.Matrix(self.dpdt_vect_s))[0, 0]

    def make_numerical_params(self,
                       d_base: float=1.0,
                       n_base: float=3.0,
                       beta_base: float=2.0,
                       k_base: float = 10.0,
                       mu_base: float=0.5) -> None:
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
                              constraint_inds: list|None = None,
                              constraint_vals: list|None = None):
        '''

        '''
        # First, lambdify the change vector in a way that supports any constraints:
        if constraint_inds is None or constraint_vals is None:
            if self._inter_funk_type is InterFuncType.hill:
                dpdt_vect_f = sp.lambdify((self.p_vect_s,
                                           np.asarray(self.M_n_s),
                                           np.asarray(self.M_beta_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)
            elif self._inter_funk_type is InterFuncType.logistic:
                dpdt_vect_f = sp.lambdify((self.p_vect_s,
                                           np.asarray(self.M_k_s),
                                           np.asarray(self.M_mu_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)

            else:
                raise Exception("Only InterFuncType hill and logistic are supported.")

        else: # If there are constraints split the p-vals into an arguments and to-solve set:
            p_vect_args = np.asarray(self.p_vect_s)[constraint_inds]
            p_vect_solve = np.setdiff1d(self.p_vect_s, p_vect_args)

            if self._inter_funk_type is InterFuncType.hill:
                dpdt_vect_f = sp.lambdify((p_vect_solve,
                                           p_vect_args,
                                           np.asarray(self.M_n_s),
                                           np.asarray(self.M_beta_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)

            elif self._inter_funk_type is InterFuncType.logistic:
                dpdt_vect_f = sp.lambdify((p_vect_solve,
                                           p_vect_args,
                                           np.asarray(self.M_k_s),
                                           np.asarray(self.M_mu_s),
                                           self.d_vect_s),
                                          self.dpdt_vect_s)

            else:
                raise Exception("Only InterFuncType hill and logistic are supported.")

        return dpdt_vect_f

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
                                constrained_inds: list | None = None,
                                constrained_vals: list | None = None,
                                d_base: float = 1.0,
                                n_base: float = 3.0,
                                beta_base: float = 2.0,
                                k_base: float = 10.0,
                                mu_base: float = 0.5,
                                N_space: int = 2,
                                pmin: float=1.0e-25,
                                tol: float=1.0e-15,
                                N_round_sol: int = 2,
                                ):
        '''
        Solve for the equilibrium points of gene product probabilities in
        terms of a given set of numerical parameters.
        '''

        dpdt_vect_f = self.create_numerical_dpdt(constraint_inds=constrained_inds,
                                                 constraint_vals=constrained_vals)

        self.make_numerical_params(d_base=d_base,
                                   n_base=n_base,
                                   beta_base=beta_base,
                                   k_base=k_base,
                                   mu_base=mu_base)

        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self._gmod.nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self._gmod.nodes_index, constrained_inds)

        M_pstates = self.generate_state_space(unconstrained_inds, N_space, pmin)

        sol_Mo = []

        for pvecto in M_pstates: # for each test vector:
            p_vect_sol = pvecto[unconstrained_inds] # get values for the genes we're solving for...

            # function_args = (constraint_vals, self.M_n, self.M_beta, self.d_vect)
            function_args = self.get_function_args(constraint_vals=constrained_vals)

            sol_roots = fsolve(dpdt_vect_f, p_vect_sol, args=function_args, xtol=tol)

            p_eqms = np.zeros(self._gmod.N_nodes)
            p_eqms[unconstrained_inds] = sol_roots

            if constrained_inds is not None and constrained_vals is not None:
                p_eqms[constrained_inds] = constrained_vals

            sol_Mo.append(p_eqms)

        _, unique_inds = np.unique(np.round(sol_Mo, N_round_sol), axis=0, return_index=True)

        sol_M = np.asarray(sol_Mo)[unique_inds]

        return sol_M


