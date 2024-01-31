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
from scipy.optimize import fsolve
import sympy as sp
from sympy import MutableDenseMatrix
from cellnition.science.network_models.network_abc import NetworkABC
from cellnition.science.network_enums import EdgeType, InterFuncType, CouplingType

# FIXME: I bet we can implement a node type via the same process

class ProbabilityNet(NetworkABC):
    '''
    '''
    def __init__(self,
                 N_nodes: int,
                 interaction_function_type: InterFuncType = InterFuncType.hill):
        '''

        '''

        super().__init__()  # Initialize the base class

        self._N_nodes = N_nodes
        self._inter_fun_type = interaction_function_type
        self._nodes_index = [i for i in range(self._N_nodes)]

        # Matrix Equations:
        # Matrix symbols to construct matrix equation bases:
        self._M_n_so = sp.MatrixSymbol('M_n', self._N_nodes, self._N_nodes)
        self._M_beta_so = sp.MatrixSymbol('M_beta', self._N_nodes, self._N_nodes)
        self._M_p_so = sp.MatrixSymbol('M_p', self._N_nodes, self._N_nodes)

        # Define symbolic adjacency matrices to use as masks in defining multi and add matrices:
        self._A_add_so = sp.MatrixSymbol('A_add', N_nodes, N_nodes)
        self._A_mul_so = sp.MatrixSymbol('A_mul', N_nodes, N_nodes)

        # Now we can define symbolic matrices that use the add and mul adjacencies to mask which
        # n-parameters to use:
        M_n_add_so = sp.hadamard_product(self._A_add_so, self._M_n_so)
        M_n_mul_so = sp.hadamard_product(self._A_mul_so, self._M_n_so)

        # And functions can be plugged in as matrix equations; these are the fundamental
        # model building functions:
        if self._inter_fun_type is InterFuncType.hill:
            # For Hill Functions:
            self._M_funk_add_so = sp.Matrix(N_nodes, N_nodes,
                                      lambda i, j: 1 / (1 + (self._M_beta_so[j, i] * self._M_p_so[j, i]) ** -M_n_add_so[j, i]))
            self._M_funk_mul_so = sp.Matrix(N_nodes, N_nodes,
                                      lambda i, j: 1 / (1 + (self._M_beta_so[j, i] * self._M_p_so[j, i]) ** -M_n_mul_so[j, i]))
        else:
            # For Logistic Functions:
            self._M_funk_add_so = sp.Matrix(N_nodes, N_nodes,
                                      lambda i, j: 1 / (1 + sp.exp(-M_n_add_so[j, i] * (self._M_p_so[j, i] -
                                                                                        self._M_beta_so[j, i]))))
            self._M_funk_mul_so = sp.Matrix(N_nodes, N_nodes,
                                      lambda i, j: 1 / (1 + sp.exp(-M_n_mul_so[j, i] * (self._M_p_so[j, i] -
                                                                                        self._M_beta_so[j, i]))))

        # Symbolic model parameters:
        self._d_s = sp.IndexedBase('d', shape=self._N_nodes, positive=True)  # Maximum rate of decay
        self._p_s = sp.IndexedBase('p', shape=self._N_nodes, positive=True)  # Probability of gene product

        # Vectorized node-parameters and variables:
        self._d_vect_s = [self._d_s[i] for i in range(self._N_nodes)]  # maximum rate of decay for each node
        self._c_vect_s = sp.Matrix([self._p_s[i] for i in range(self._N_nodes)])  # gene product probability for each node

        self._beta_s = sp.IndexedBase('beta', shape=(self._N_nodes, self._N_nodes), positive=True)  # Hill centre
        self._n_s = sp.IndexedBase('n', shape=(self._N_nodes, self._N_nodes), positive=True)  # Hill coupling

        # Create a matrix out of the n_s symbols:
        self._M_n_s = sp.Matrix(self._N_nodes, self._N_nodes,
                                lambda i, j: self._n_s[i, j])

        self._M_beta_s = sp.Matrix(self._N_nodes, self._N_nodes,
                                   lambda i, j: self._beta_s[i, j])

        # Define vector of ones to use in matrix operations:
        self._ones_vect = sp.ones(1, self._N_nodes)

        # Create a matrix that allows us to access the concentration vectors
        # duplicated along columns:
        self._M_p_s = self._c_vect_s * self._ones_vect

    def build_adjacency_from_edge_type_list(self,
                        edge_types: list[EdgeType],
                        edges_index: list[tuple[int,int]],
                        coupling_type: CouplingType=CouplingType.specified):
        '''

        '''
        A_full_s = np.zeros((self._N_nodes, self._N_nodes), dtype=int)
        A_add_s = np.zeros((self._N_nodes, self._N_nodes), dtype=int)
        A_mul_s = np.zeros((self._N_nodes, self._N_nodes), dtype=int)

        # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
        # and multiplicative interactions:
        for ei, ((nde_i, nde_j), etype) in enumerate(zip(edges_index, edge_types)):
            if etype is EdgeType.A or etype is EdgeType.As:
                A_full_s[nde_i, nde_j] = 1
            elif etype is EdgeType.I or etype is EdgeType.Is:
                A_full_s[nde_i, nde_j] = -1

        for ei, ((nde_i, nde_j), etype) in enumerate(zip(edges_index, edge_types)):
            if coupling_type is CouplingType.specified:
                if etype is EdgeType.I:
                    A_add_s[nde_i, nde_j] = -1
                elif etype is EdgeType.A:
                    A_add_s[nde_i, nde_j] = 1
                elif etype is EdgeType.Is:
                    A_mul_s[nde_i, nde_j] = -1
                elif etype is EdgeType.As:
                    A_mul_s[nde_i, nde_j] = 1

            elif coupling_type is CouplingType.additive:
                if etype is EdgeType.I or etype is EdgeType.Is:
                    A_add_s[nde_i, nde_j] = -1
                elif etype is EdgeType.A or etype is EdgeType.As:
                    A_add_s[nde_i, nde_j] = 1

            elif coupling_type is CouplingType.multiplicative:
                if etype is EdgeType.I or etype is EdgeType.Is:
                    A_mul_s[nde_i, nde_j] = -1
                elif etype is EdgeType.A or etype is EdgeType.As:
                    A_mul_s[nde_i, nde_j] = 1

            elif coupling_type is CouplingType.mixed:
                if etype is EdgeType.A or etype is EdgeType.As:
                    A_add_s[nde_i, nde_j] = 1
                elif etype is EdgeType.I or etype is EdgeType.Is:
                    A_mul_s[nde_i, nde_j] = -1

        A_add_s = sp.Matrix(A_add_s)
        A_mul_s = sp.Matrix(A_mul_s)
        A_full_s = sp.Matrix(A_full_s)

        return A_add_s, A_mul_s, A_full_s

    def get_adjacency_randomly(self, coupling_type: CouplingType=CouplingType.mixed, set_autoactivation: bool=True):
        '''
        Return a randomly-generated full adjacency matrix.
        '''
        A_full_s = sp.Matrix(np.random.randint(-1, 2, size=(self._N_nodes, self._N_nodes)))

        if set_autoactivation:
            # Make it so that any diagonal elements are self-activating rather than self-inhibiting
            A_full_s = sp.Matrix(self._N_nodes, self._N_nodes,
                                 lambda i,j: A_full_s[i,j]*A_full_s[i,j] if i==j else A_full_s[i,j])

        A_add_s, A_mul_s = self.process_full_adjacency(A_full_s, coupling_type=coupling_type)

        return A_add_s, A_mul_s, A_full_s


    def edges_from_adjacency(self, A_add_s: MutableDenseMatrix, A_mul_s: MutableDenseMatrix):
        '''
        Returns edge type and edge index from adjacency matrices.
        '''
        edges_type = []
        edges_index = []

        A_full_s = A_add_s + A_mul_s
        for i in range(self._N_nodes):
            for j in range(self._N_nodes):
                afull_ij = A_full_s[i,j]
                if afull_ij != 0:
                    edges_index.append((i, j))
                    if A_add_s[i,j] < 0:
                        edges_type.append(EdgeType.I)
                    elif A_add_s[i,j] > 0:
                        edges_type.append(EdgeType.A)
                    elif A_mul_s[i,j] < 0:
                        edges_type.append(EdgeType.Is)
                    elif A_mul_s[i,j] > 0:
                        edges_type.append(EdgeType.As)
                    else:
                        edges_type.append(EdgeType.N)

        return edges_index, edges_type


    def process_full_adjacency(self, A_full_s: MutableDenseMatrix, coupling_type: CouplingType=CouplingType.mixed):
        '''
        Process a full adjacency matrix into additive and multiplicative components
        based on a specified coupling type.

        '''
        A_add_s = np.zeros((self._N_nodes, self._N_nodes), dtype=int)
        A_mul_s = np.zeros((self._N_nodes, self._N_nodes), dtype=int)

        for i in range(self._N_nodes):
            for j in range(self._N_nodes):
                afull_ij = A_full_s[i,j]
                if afull_ij == 1:
                    if coupling_type is CouplingType.additive or coupling_type is CouplingType.mixed:
                        A_add_s[i,j] = 1
                    elif coupling_type is CouplingType.multiplicative:
                        A_mul_s[i,j] = 1
                    else:
                        raise Exception('CouplingType.specified is not supported in this method.')

                if afull_ij == -1:
                    if coupling_type is CouplingType.additive:
                        A_add_s[i,j] = -1
                    elif coupling_type is CouplingType.multiplicative or coupling_type is CouplingType.mixed:
                        A_mul_s[i,j] = -1
                    else:
                        raise Exception('CouplingType.specified is not supported in this method.')

        A_add_s = sp.Matrix(A_add_s)
        A_mul_s = sp.Matrix(A_mul_s)

        return A_add_s, A_mul_s


    def build_analytical_model(self,
                 A_add_s: MutableDenseMatrix,
                 A_mul_s: MutableDenseMatrix
                               ):
        '''

        '''

        # Initialize a list of node indices that should be constrained (removed from solution searches)
        # due to their lack of regulation:
        self._constrained_nodes = []

        if A_add_s.shape != (self._N_nodes, self._N_nodes):
            raise Exception("Shape of A_add_s is not in terms of network node number!")

        if A_mul_s.shape != (self._N_nodes, self._N_nodes):
            raise Exception("Shape of A_add_s is not in terms of network node number!")

        M_funk_add_si = self._M_funk_add_so.subs(
            [(self._M_p_so, self._M_p_s),
             (self._M_n_so, self._M_n_s),
             (self._M_beta_so, self._M_beta_s),
             (self._A_add_so, A_add_s)])

        M_funk_mul_si = self._M_funk_mul_so.subs(
            [(self._M_p_so, self._M_p_s),
             (self._M_n_so, self._M_n_s),
             (self._M_beta_so, self._M_beta_s),
             (self._A_mul_so, A_mul_s)])

        # Filter out the 1/2 terms and set to 0 (addiditive) or 1 (multiplicative):
        M_funk_add_s = sp.Matrix(self._N_nodes, self._N_nodes, lambda i, j: sp.Piecewise(
            (M_funk_add_si[i, j], M_funk_add_si[i, j] != sp.Rational(1, 2)),
            (0, True)))

        M_funk_mul_s = sp.Matrix(self._N_nodes, self._N_nodes, lambda i, j: sp.Piecewise(
            (M_funk_mul_si[i, j], M_funk_mul_si[i, j] != sp.Rational(1, 2)),
            (1, True)))

        # As A_add_s is a signed adjacency matrix, we need to get the absolute value to count edges:
        abs_A_add_s = sp.hadamard_product(A_add_s, A_add_s)

        # Count the nodes interacting (on input) with each node:
        n_add_edges_i = abs_A_add_s.T * self._ones_vect.T
        # Correct for any zeros in the n_add_edges and create a normalization object:
        self._n_add_edges = sp.Matrix(self._N_nodes, 1,
                                lambda i, j: sp.Piecewise((sp.Rational(1, n_add_edges_i[i, j]), n_add_edges_i[i, j] != 0),
                                                          (1, True)))
        add_terms_i = M_funk_add_s * self._ones_vect.T

        # The add_terms need to be normalized to keep concentrations between 0 and 1:
        self._add_terms = sp.hadamard_product(self._n_add_edges, add_terms_i)

        self._mul_terms = sp.Matrix(np.product(M_funk_mul_s, axis=1))

        self._dcdt_vect_s = []
        for i in range(self._N_nodes):
            if self._add_terms[i] == 0 and self._mul_terms[i] == 1: # if there's no add term and no mul term
                self._dcdt_vect_s.append(0) # set the rate of change of this unregulated node to zero
                self._constrained_nodes.append(i) # append this node to the list of nodes that should be constrained
            elif self._add_terms[i] == 0 and self._mul_terms[i] != 1: # remove the null add term to avoid nulling all growth
                self._dcdt_vect_s.append(self._d_vect_s[i] * self._mul_terms[i] -
                                         self._c_vect_s[i] * self._d_vect_s[i])
            else: # the node is a mix of additive and potential multiplicative regulation:
                self._dcdt_vect_s.append(self._d_vect_s[i] * self._mul_terms[i] * self._add_terms[i] -
                                         self._c_vect_s[i] * self._d_vect_s[i])

        # This is an "energy" function to be minimized at the equilibrium points:
        self._opti_s = (sp.Matrix(self._dcdt_vect_s).T * sp.Matrix(self._dcdt_vect_s))[0, 0]

        # Create linearized lists of symbolic parameters that are needed to solve the model (exclude the
        # zero entries of the M_n and M_beta matrices:
        # FIXME: need to rebuild the graph model if edges index changes...
        self._edges_index, self.edge_types = self.edges_from_adjacency(A_add_s, A_mul_s)
        self._beta_vect_s = [self._beta_s[nde_i, nde_j] for nde_i, nde_j in self._edges_index]
        self._n_vect_s = [self._n_s[nde_i, nde_j] for nde_i, nde_j in self._edges_index]
        self._N_edges = len(self._edges_index) # count number of edges

    def make_numerical_params(self,
                       d_base: float|list[float]=1.0,
                       n_base: float|list[float]=3.0,
                       beta_base: float|list[float]=2.0,
                       ) -> tuple[list[float], list[float], list[float]]:
        '''
        Scrape the network for base parameters to initialize numerical parameters.

        '''
        # Node parameters:
        if type(d_base) is list:
            assert len(d_base) == self._N_nodes, "Length of d_base not equal to node number!"
            d_vect = d_base
        else:
            d_vect = [d_base for i in range(self._N_nodes)]

        # Edge parameters:
        if type(n_base) is list:
            assert len(n_base) == self._N_edges, "Length of n_base not equal to edge number!"
            n_vect = n_base
        else:
            n_vect = [n_base for i in range(self._N_edges)]

        if type(beta_base) is list:
            assert len(beta_base) == self._N_edges, "Length of n_base not equal to edge number!"
            beta_vect = beta_base
        else:
            beta_vect = [beta_base for i in range(self._N_edges)]

        return d_vect, n_vect, beta_vect

    def create_numerical_dcdt(self,
                              constrained_inds: list | None = None,
                              constrained_vals: list | None = None):
        '''

        '''
        # First, lambdify the change vector in a way that supports any constraints:
        if constrained_inds is None or constrained_vals is None:
            # Compute the symbolic Jacobian:
            dcdt_jac_s = sp.Matrix(self._dcdt_vect_s).jacobian(self._c_vect_s) # analytical Jacobian

            dcdt_vect_f = sp.lambdify((list(self._c_vect_s),
                                       self._n_vect_s,
                                       self._beta_vect_s,
                                       self._d_vect_s),
                                      self._dcdt_vect_s)

            dcdt_jac_f = sp.lambdify((list(self._c_vect_s),
                                      self._n_vect_s,
                                      self._beta_vect_s,
                                      self._d_vect_s),
                                     dcdt_jac_s)


        else: # If there are constraints split the p-vals into an arguments and to-solve set:
            c_vect_args = (np.asarray(list(self._c_vect_s))[constrained_inds]).tolist()
            unconstrained_inds = np.setdiff1d(self._nodes_index, constrained_inds).tolist()
            c_vect_solve = (np.asarray(list(self._c_vect_s))[unconstrained_inds]).tolist()

            # truncate the change vector to only be for unconstrained inds:
            dcdt_vect_s = np.asarray(self._dcdt_vect_s)[unconstrained_inds].tolist()

            # Compute the symbolic Jacobian:
            dcdt_jac_s = sp.Matrix(dcdt_vect_s).jacobian(c_vect_solve) # analytical Jacobian

            dcdt_vect_f = sp.lambdify((c_vect_solve,
                                       c_vect_args,
                                       self._n_vect_s,
                                       self._beta_vect_s,
                                       self._d_vect_s),
                                      dcdt_vect_s)

            dcdt_jac_f = sp.lambdify((c_vect_solve,
                                      c_vect_args,
                                      self._n_vect_s,
                                      self._beta_vect_s,
                                      self._d_vect_s),
                                     dcdt_jac_s)

        return dcdt_vect_f, dcdt_jac_f

    def get_function_args(self,
                          constraint_vals: list|None=None,
                          d_base: float|list[float]=1.0,
                          n_base: float|list[float]=3.0,
                          beta_base: float|list[float]=2.0):
        '''

        '''
        d_vect, n_vect, beta_vect = self.make_numerical_params(d_base, n_base, beta_base)

        if constraint_vals is not None:
            function_args = (constraint_vals, n_vect, beta_vect, d_vect)

        else:
            function_args = (n_vect, beta_vect, d_vect)

        return function_args


    def generate_state_space(self,
                             c_inds: list[int],
                             N_space: int,
                             pmin: float=1.0e-25) -> tuple[ndarray, list, ndarray]:
        '''
        Generate a discrete state space over the range of probabilities of
        each individual gene in the network.
        '''
        c_lins = []

        for i in c_inds:
            c_lins.append(np.linspace(pmin, 1.0, N_space))

        cGrid = np.meshgrid(*c_lins)

        N_pts = len(cGrid[0].ravel())

        cM = np.zeros((N_pts, self._N_nodes))

        for i, cGrid in enumerate(cGrid):
            cM[:, i] = cGrid.ravel()

        return cM, c_lins, cGrid

    def solve_probability_equms(self,
                                constrained_inds: list|None = None,
                                constrained_vals: list|None = None,
                                d_base: float = 1.0,
                                n_base: float = 3.0,
                                beta_base: float = 4.0,
                                N_space: int = 2,
                                pmin: float=1.0e-9,
                                search_tol: float=1.0e-15,
                                sol_tol: float=1.0e-1,
                                N_round_sol: int = 2,
                                verbose: bool=True,
                                save_file: str|None = None
                                ):
        '''
        Solve for the equilibrium points of gene product probabilities in
        terms of a given set of numerical parameters.
        '''

        dcdt_vect_f, dcdt_jac_f = self.create_numerical_dcdt(constrained_inds=constrained_inds,
                                                             constrained_vals=constrained_vals)

        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self._nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self._nodes_index, constrained_inds).tolist()

        M_pstates, _, _ = self.generate_state_space(unconstrained_inds, N_space, pmin)

        sol_Mo = []

        function_args = self.get_function_args(constraint_vals=constrained_vals,
                                               d_base=d_base,
                                               n_base=n_base,
                                               beta_base=beta_base)

        for cvecto in M_pstates: # for each test vector:
            c_vect_sol = cvecto[unconstrained_inds] # get values for the genes we're solving for...

            sol_roots = fsolve(dcdt_vect_f,
                               c_vect_sol,
                               args=function_args,
                               xtol=search_tol,
                               fprime=dcdt_jac_f,
                               col_deriv=False,
                               )

            c_eqms = np.zeros(self._N_nodes)
            c_eqms[unconstrained_inds] = sol_roots

            if constrained_inds is not None and constrained_vals is not None:
                c_eqms[constrained_inds] = constrained_vals

            sol_Mo.append(c_eqms)

        _, unique_inds = np.unique(np.round(sol_Mo, N_round_sol), axis=0, return_index=True)

        sol_M = np.asarray(sol_Mo)[unique_inds]

        stable_sol_M, sol_M_char = self._find_attractor_sols(sol_M,
                                                             dcdt_vect_f,
                                                             dcdt_jac_f,
                                                             function_args,
                                                             constrained_inds=constrained_inds,
                                                             tol= sol_tol,
                                                             verbose = verbose,
                                                             unique_sols = True,
                                                             sol_round = N_round_sol,
                                                             save_file = save_file)

        return stable_sol_M, sol_M_char, sol_M
    def _find_attractor_sols(self,
                             sols_0: list|ndarray,
                             dcdt_vect_f: Callable,
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

        eps = 1.0e-25 # Small value to avoid divide-by-zero in the jacobian

        sol_dicts_list = []

        if constrained_inds is None:
            unconstrained_inds = self._nodes_index

        else:
            unconstrained_inds = np.setdiff1d(self._nodes_index, constrained_inds)

        for pminso in sols_0:

            solution_dict = {}

            solution_dict['Minima Values'] = pminso

            pmins = np.asarray(pminso) + eps # add the small amount here, before calculating the jacobian

            solution_dict['Change at Minima'] = dcdt_vect_f(pmins[unconstrained_inds], *func_args)

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

        return solsM_return, sol_dicts_list

    def run_time_sim(self,
                     tend: float,
                     dt: float,
                     cvecti: ndarray|list,
                     sig_inds: ndarray|list|None = None,
                     sig_times: ndarray | list | None = None,
                     sig_mag: ndarray | list | None = None,
                     dt_samp: float|None = None,
                     constrained_inds: list | None = None,
                     constrained_vals: list | None = None,
                     d_base: float = 1.0,
                     n_base: float = 3.0,
                     beta_base: float = 4.0
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
        dtv = 1.0e-3 * np.ones(self._N_nodes)
        dtv[self.sensor_node_inds] = 1.0

        dcdt_vect_f, dcdt_jac_f = self.create_numerical_dcdt(constrained_inds=constrained_inds,
                                                             constrained_vals=constrained_vals)

        function_args = self.get_function_args(constraint_vals=constrained_vals,
                                               d_base=d_base,
                                               n_base=n_base,
                                               beta_base=beta_base)

        for ti, tt in enumerate(tvect):
            dcdt = dcdt_vect_f(cvecti, *function_args)
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


