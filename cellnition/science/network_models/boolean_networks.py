#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds a Boolean network model from a directed graph. The
module is founded on the baseclass NetworkABC, so can generate random
directed graphs, load them from edges, and perform graph analysis.

'''
import csv
import itertools
from collections.abc import Callable
from cellnition.science.network_models.network_abc import NetworkABC
import numpy as np
from numpy import ndarray
import sympy as sp
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             CouplingType,
                                                             EquilibriumType)
from cellnition.types import NumpyTrue
import networkx as nx

class BooleanNet(NetworkABC):
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
    def __init__(self):
        '''

        '''

        super().__init__()  # Initialize the base class

        # Init all core attributes:
        self.edges_list = None
        self.nodes_list = None
        self.GG = None
        self.N_edges = None
        self.N_nodes = None
        self.edges_index = None
        self.nodes_index = None

        self._c_vect_s = None
        self._A_bool_s = None
        self._A_bool_f = None

    def build_adjacency_matrices(self,
                        edge_types: list[EdgeType],
                        edges_index: list[tuple[int,int]],
                        ):
        '''

        '''
        # Initialize an activator matrix:
        A_acti_s = np.zeros((self.N_nodes, self.N_nodes), dtype=int)
        # Initialize an inhibitor matrix:
        A_inhi_s = np.zeros((self.N_nodes, self.N_nodes), dtype=int)

        # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
        # and multiplicative interactions:cc
        for ei, ((nde_i, nde_j), etype) in enumerate(zip(edges_index, edge_types)):
            if etype is EdgeType.A or etype is EdgeType.As:
                A_acti_s[nde_j, nde_i] = 1
            elif etype is EdgeType.I or etype is EdgeType.Is:
                A_inhi_s[nde_j, nde_i] = 1

        A_acti_s = sp.Matrix(A_acti_s)
        A_inhi_s = sp.Matrix(A_inhi_s)

        return A_acti_s, A_inhi_s


    #----Boolean Model Building--------
    def build_boolean_model(self, use_node_name: bool=True,
                            multi_coupling_type: CouplingType=CouplingType.mix1,
                            constitutive_express: bool = False):
        '''
        Construct a Boolean solver for a network.Returns both the symbolic equations (A_bool_s)
        as well as a vectorized numpy function (A_bool_f) that accepts the list or array of
        node concentrations.

        Parameters
        ----------
        use_node_name: bool = True
            If True, the node label is used as the symbolic parameter name. Otherwise a shorter
            parameter label of 'g_#" is used, where # is the node's index in the network.

        multi_coupling_type: CouplingType=CouplingType.mixed
            Specify how factors are combined at individual target nodes.


        '''
        if use_node_name:
            c_vect_s = sp.Matrix([sp.Symbol(nde_nme, positive=True) for nde_nme in self.nodes_list])
        else: # use the node's numerical index as a label
            c_vect_s = sp.Matrix([sp.Symbol(f'g_{nde_i}',
                                            positive=True) for nde_i in self.nodes_index])

        if multi_coupling_type is CouplingType.mix1:
            # Case #1: Mixed coupling, where inhibitors always act in "OR"
            # configuration and activators act in "AND" configuration.
            # Initialize an activator matrix:
            A_acti_so = np.zeros((self.N_nodes, self.N_nodes), dtype=int)
            # onesv = np.ones(self.N_nodes, dtype=int)

            # Initialize an inhibitor matrix:
            A_inhi_so = np.ones((self.N_nodes, self.N_nodes), dtype=sp.Symbol)

            acti_count = [0 for i in range(self.N_nodes)]  # counts the number of activators acting on each node
            inhi_count = [0 for i in range(self.N_nodes)]  # counts the number of inhibitors acting on each node

            # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
            # and multiplicative interactions:cc
            for ei, ((nde_i, nde_j), etype) in enumerate(zip(self.edges_index, self.edge_types)):
                # print(type(nde_i))
                # print(c_vect_s[nde_i])
                if etype is EdgeType.A or etype is EdgeType.As:
                    A_acti_so[nde_j, nde_i] = 1
                    acti_count[nde_j] += 1
                elif etype is EdgeType.I or etype is EdgeType.Is:
                    A_inhi_so[nde_j, nde_i] = 1 - c_vect_s[nde_i]
                    inhi_count[nde_j] += 1

            # Combine so that presence of activators AND absence of inhibitors required for node expressions:
            if constitutive_express is False:
                # Need to create a normalized vector for managing cooperativity of the "OR"
                denom = np.asarray(acti_count)  # total number of activators at each node
                idenom = (denom == 0).nonzero()[0]  # indices where denom is zero
                denom[idenom] = 1  # set those equal to 1
                denom = np.int64(denom)
                coopv = np.asarray([sp.Rational(1, di) for di in denom])

                A_acti_so = (coopv * A_acti_so.T).T # multiply by the normalizing vector coopv
                A_acti_ss = A_acti_so.dot(c_vect_s)

                const_inds = [] # if there's inhibitor but no activator, node must be const expressed
                for ndei, (act, ict) in enumerate(zip(acti_count, inhi_count)):
                    if act == 0 and ict != 0:
                        const_inds.append(ndei)

                A_acti_ss[const_inds] = 1 # set this to 1 where the const expressed nodes should me

                A_acti_s = sp.Matrix(A_acti_ss) # collect terms into "OR" activators at each node
                A_inhi_s = sp.Matrix(np.prod(A_inhi_so, axis=1)) # collect terms into "AND" inhibitors at each node
                A_bool_s = sp.hadamard_product(A_acti_s, A_inhi_s) # Use "AND" to combine acti and inhi

            # We use and additive "OR" to specify the presence of an activator OR absence of an inhibitor
            # is required for gene expression for all genes
            else:
                # Need to create a normalized vector for managing cooperativity of the "OR"
                # sums the number of activators and if inhibitors at each node:
                denom = np.asarray(acti_count) + np.sign(inhi_count)
                idenom = (denom == 0).nonzero()[0]  # indices where denom is zero
                denom[idenom] = 1  # set those equal to 1
                denom = np.int64(denom)
                coopv = sp.Matrix([sp.Rational(1, di) for di in denom]) # write as fractions for pretty display

                # Multiply the system through with the normalizing coefficients:
                A_acti_s = sp.hadamard_product(coopv, sp.Matrix(A_acti_so.dot(c_vect_s)))
                A_inhi_s = sp.hadamard_product(coopv, sp.Matrix(np.sign(inhi_count) * np.prod(A_inhi_so, axis=1)))
                # Combine activators and inhibitors as "OR" function:
                A_bool_s = A_acti_s + A_inhi_s


        elif multi_coupling_type is CouplingType.mix2:
            # Mixed coupling #2, where inhibitors always act in "AND"
            # configuration and activators act in "OR" configuration.
            # Initialize an inhibitor matrix:
            A_inhi_so = np.zeros((self.N_nodes, self.N_nodes), dtype=int)
            onesv = np.ones(self.N_nodes, dtype=int)

            acti_count = [0 for i in range(self.N_nodes)]  # counts the number of activators acting on each node
            inhi_count = [0 for i in range(self.N_nodes)]  # counts the number of inhibitors acting on each node

            # Initialize an activator matrix:
            A_acti_so = np.ones((self.N_nodes, self.N_nodes), dtype=sp.Symbol)

            # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
            # and multiplicative interactions:cc
            for ei, ((nde_i, nde_j), etype) in enumerate(zip(self.edges_index, self.edge_types)):
                # print(type(nde_i))
                # print(c_vect_s[nde_i])
                if etype is EdgeType.A or etype is EdgeType.As:
                    A_acti_so[nde_j, nde_i] = c_vect_s[nde_i]
                    acti_count[nde_j] += 1
                elif etype is EdgeType.I or etype is EdgeType.Is:
                    A_inhi_so[nde_j, nde_i] = 1
                    inhi_count[nde_j] += 1

            # Combine so that presence of activators AND absence of inhibitors required for node expressions:
            if constitutive_express is False:
                # Need to create a normalized vector for managing cooperativity of the "OR"
                denom = np.asarray(inhi_count)  # total number of activators at each node
                idenom = (denom == 0).nonzero()[0]  # indices where denom is zero
                denom[idenom] = 1  # set those equal to 1
                denom = np.int64(denom)
                coopv = np.asarray([sp.Rational(1, di) for di in denom])

                A_inhi_so = (coopv * A_inhi_so.T).T  # multiply by the normalizing vector coopv
                A_inhi_ss = A_inhi_so.dot(sp.Matrix(onesv) - c_vect_s)

                const_inds = []  # if there's inhibitor but no activator, node must be const expressed
                for ndei, (act, ict) in enumerate(zip(acti_count, inhi_count)):
                    if act != 0 and ict == 0:
                        const_inds.append(ndei)

                A_inhi_ss[const_inds] = 1  # set this to 1 where the const expressed nodes should me

                A_inhi_s = sp.Matrix(A_inhi_ss)  # collect terms into "OR" activators at each node
                A_acti_s = sp.Matrix(
                    np.prod(A_acti_so, axis=1))  # collect terms into "AND" inhibitors at each node
                A_bool_s = sp.hadamard_product(A_acti_s, A_inhi_s)  # Use "AND" to combine acti and inhi

            # We use and additive "OR" to specify the presence of an activator OR absence of an inhibitor
            # is required for gene expression for all genes
            else:
                # Need to create a normalized vector for managing cooperativity of the "OR"
                # sums the number of activators and if inhibitors at each node:
                denom = np.asarray(inhi_count) + np.sign(acti_count)
                idenom = (denom == 0).nonzero()[0]  # indices where denom is zero
                denom[idenom] = 1  # set those equal to 1
                denom = np.int64(denom)
                coopv = sp.Matrix([sp.Rational(1, di) for di in denom])  # write as fractions for pretty display

                # Multiply the system through with the normalizing coefficients:
                A_inhi_s = sp.hadamard_product(coopv, sp.Matrix(A_inhi_so.dot(sp.Matrix(onesv) - c_vect_s)))
                A_acti_s = sp.hadamard_product(coopv,
                                               sp.Matrix(np.sign(acti_count) * np.prod(A_acti_so, axis=1)))
                # Combine activators and inhibitors as "OR" function:
                A_bool_s = A_acti_s + A_inhi_s


        elif multi_coupling_type is CouplingType.additive:

            # Case #2: Additive coupling, where all interactions inhibitors always act in
            # "AND" configuration.
            # Initialize an activator matrix:
            A_acti_so = np.zeros((self.N_nodes, self.N_nodes), dtype=int)
            # Initialize an inhibitor matrix:
            A_inhi_so = np.zeros((self.N_nodes, self.N_nodes), dtype=int)

            # Initialize a "ones vector" for each node:
            onesv = np.ones(self.N_nodes, dtype=int)

            # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
            # and multiplicative interactions:cc
            for ei, ((nde_i, nde_j), etype) in enumerate(zip(self.edges_index, self.edge_types)):
                # print(type(nde_i))
                # print(c_vect_s[nde_i])
                if etype is EdgeType.A or etype is EdgeType.As:
                    A_acti_so[nde_j, nde_i] = 1
                elif etype is EdgeType.I or etype is EdgeType.Is:
                    A_inhi_so[nde_j, nde_i] = 1

            ic_vect_s = sp.Matrix(onesv) - c_vect_s

            denom = (A_acti_so + A_inhi_so).dot(onesv)
            idenom = (denom == 0).nonzero()[0]  # indices where denom is zero
            denom[idenom] = 1  # set those equal to 1
            denom = np.int64(denom)
            # coopv = 1 / denom
            coopv = np.asarray([sp.Rational(1, di) for di in denom])

            A_acti_so = (coopv * A_acti_so.T).T
            A_inhi_so = (coopv * A_inhi_so.T).T

            A_acti_s = sp.Matrix(A_acti_so.dot(c_vect_s))
            A_inhi_s = sp.Matrix(A_inhi_so.dot(ic_vect_s))
            A_bool_s = A_acti_s + A_inhi_s

        elif multi_coupling_type is CouplingType.multiplicative:
            # Case #1: Mixed coupling, where inhibitors always act in "OR"
            # configuration and activators act in "AND" configuration.
            # Initialize an activator matrix:
            A_acti_so = np.ones((self.N_nodes, self.N_nodes), dtype=sp.Symbol)
            # Initialize an inhibitor matrix:
            A_inhi_so = np.ones((self.N_nodes, self.N_nodes), dtype=sp.Symbol)

            # Build A_full_s, an adjacency matrix that doesn't distinguish between additive
            # and multiplicative interactions:cc
            for ei, ((nde_i, nde_j), etype) in enumerate(zip(self.edges_index, self.edge_types)):
                # print(type(nde_i))
                # print(c_vect_s[nde_i])
                if etype is EdgeType.A or etype is EdgeType.As:
                    A_acti_so[nde_j, nde_i] = c_vect_s[nde_i]
                elif etype is EdgeType.I or etype is EdgeType.Is:
                    A_inhi_so[nde_j, nde_i] = 1 - c_vect_s[nde_i]

            A_acti_s = sp.Matrix(np.prod(A_acti_so, axis=1))
            A_inhi_s = sp.Matrix(np.prod(A_inhi_so, axis=1))
            A_bool_s = sp.hadamard_product(A_acti_s, A_inhi_s)

        else:
            raise Exception("Only additive, multiplicative, and mixed coupling types are supported")

        # Finally, create a vectorized numpy function to calculate the result:
        A_bool_f = sp.lambdify([c_vect_s], A_bool_s.T)
        self._c_vect_s = c_vect_s
        self._A_bool_s = A_bool_s
        self._A_bool_f = A_bool_f

        return c_vect_s, A_bool_s, A_bool_f

    #---Boolean Model Solving----------
    def net_state_compute(self,
                              cc_o: ndarray|list,
                              A_bool_f: Callable,
                              n_max_steps: int=20,
                              constraint_inds: list|None=None,
                              constraint_vals: list|None=None,
                              verbose: bool=False
                              ):
        '''

        '''
        cc_i = cc_o  # initialize the function values (node values)
        solsv = [cc_o]  # holds a list of transient solutions
        sol_char = EquilibriumType.undetermined # initialize to undetermined

        for i in range(n_max_steps):

            if verbose:
                print(solsv[-1])
            # A true "OR" function will return the maximum of the list of booleans. This can
            # be achieved by using the "ceiling" function. If cooperative interaction is
            # desired, then rounding is better

            cc_i = np.sign(A_bool_f(cc_i)[0])  # calculate new state values

            # If there are constraints on some node vals, force them to the constraint:
            if constraint_inds is not None and constraint_vals is not None:
                cc_i[constraint_inds] = constraint_vals

            solsv.append(cc_i)

            # Detect whether we're at a steady-state:
            if (solsv[i] == solsv[i - 1]).all() is NumpyTrue:
                sol_char = EquilibriumType.attractor
                break

            elif i == n_max_steps -1:
                # test to see if we have a more complicated repetition motif:
                solvr = np.asarray(solsv)[:, self.noninput_node_inds] # get the reduced array
                si = solvr[-1, :] # try selecting the last state to check for repetition...
                matched_inds = [i for i, x in enumerate(solvr.tolist()) if x == si.tolist()] # look for repetition
                if len(matched_inds) > 1: # if there's more than one incidence of the state
                    # motif_period = matched_inds[-1] - matched_inds[-2]
                    # solvrr = solvr[3:, :] # remove the first three states...
                    # solvrr = solvr
                    # test_repeat = np.roll(solvrr, motif_period - 1, axis=1) - solvrr # roll to make the arrays equivalent

                    # if (np.sum(test_repeat) == 0):
                    motif = np.asarray(solsv)[matched_inds[-2]:matched_inds[-1], :] # extract a motif from the full array
                    cc_i = np.mean(motif, axis=0) # solution becomes the (non-integer!) mean of the motif
                    if len(motif) > 2:
                        sol_char = EquilibriumType.limit_cycle
                    else: # otherwise the motif is a saddle (metabstable point):
                        sol_char = EquilibriumType.saddle

        # self._solsv = solsv # save this so we can investigate the solution

        return cc_i, sol_char

    def net_sequence_compute(self,
                              cc_o: ndarray|list,
                              A_bool_f: Callable,
                              n_max_steps: int=20,
                              constraint_inds: list|None=None,
                              constraint_vals: list|None=None,
                              verbose: bool=False
                              ):
        '''
        Returns the sequence of states occuring after an initial state, cc_o.
        '''
        cc_i = cc_o  # initialize the function values (node values)
        solsv = [np.asarray(cc_o)]  # holds a list of transient solutions
        sol_char = EquilibriumType.undetermined # initialize to undetermined

        motif = None

        for i in range(n_max_steps):

            if verbose:
                print(solsv[-1])
            # A true "OR" function will return the maximum of the list of booleans. This can
            # be achieved by using the "ceiling" function. If cooperative interaction is
            # desired, then rounding is better

            cc_i = np.sign(A_bool_f(cc_i)[0])  # calculate new state values

            # If there are constraints on some node vals, force them to the constraint:
            if constraint_inds is not None and constraint_vals is not None:
                cc_i[constraint_inds] = constraint_vals

            solsv.append(cc_i)

            if i == n_max_steps -1: # ready to characterize the array
                # Detect whether we're at a steady-state:
                if (solsv[i] == solsv[i - 1]).all() is NumpyTrue:
                    sol_char = EquilibriumType.attractor
                    motif = solsv[i]

                else:
                    # test to see if we have a more complicated repetition motif:
                    solvr = np.asarray(solsv)[:, self.noninput_node_inds] # get the reduced array
                    si = solvr[-1, :] # try selecting the last state to check for repetition...
                    matched_inds = [i for i, x in enumerate(solvr.tolist()) if x == si.tolist()] # look for repetition
                    if len(matched_inds) > 1: # if there's more than one incidence of the state
                        motif_period = matched_inds[-1] - matched_inds[-2]
                        # solvrr = solvr[3:, :] # remove the first three states...
                        solvrr = solvr
                        test_repeat = np.roll(solvrr, motif_period - 1, axis=1) - solvrr # roll to make the arrays equivalent

                        if (np.sum(test_repeat) == 0):
                            motif = np.asarray(solsv)[matched_inds[-2]:matched_inds[-1], :] # extract a motif from the full array
                            cc_i = np.mean(motif, axis=0) # solution becomes the (non-integer!) mean of the motif
                            if len(motif) > 2:
                                sol_char = EquilibriumType.limit_cycle
                            else: # otherwise the motif is a saddle (metabstable point):
                                sol_char = EquilibriumType.saddle

        return solsv, cc_i, sol_char, motif

    def net_multisequence_compute(self,
                                  cc_o: ndarray|list,
                                  sigs_vect: ndarray|list,
                                  A_bool_f: Callable,
                                  n_max_steps: int=20,
                                  constraint_inds: list|None=None,
                                  verbose: bool=False):
        '''

        '''
        sequence_results = []
        sol_results = []
        sol_char_results = []

        for sig_vals in sigs_vect:
            cc_o[constraint_inds] = sig_vals
            solsv, cc_o, sol_char, motif = self.net_sequence_compute(cc_o,
                                                                     A_bool_f,
                                                                     n_max_steps=n_max_steps,
                                                                     constraint_inds=constraint_inds,
                                                                     constraint_vals=sig_vals,
                                                                     verbose=verbose
                                                                     )
            sequence_results.append(solsv)
            sol_results.append(cc_o)
            sol_char_results.append(sol_char)

        return sequence_results, sol_results, sol_char_results




    def solve_system_equms(self,
                           A_bool_f: Callable,
                           constraint_inds: list|None = None,
                           constraint_vals: list|None = None,
                           signal_constr_vals: list|None = None,
                           search_main_nodes_only: bool=False,
                           n_max_steps: int = 20,
                           verbose: bool=False,
                           node_num_max: int|None=None,
                           ):
        '''
        Solve for the equilibrium states of gene product in
        terms of a given set of boolean (0, 1) values.
        '''

        # For any network, there may be nodes without regulation that require constraints
        # (these are in self._constrained_nodes). Therefore, add these to any user-supplied
        # constraints:
        constrained_inds, constrained_vals = self._handle_constrained_nodes(constraint_inds,
                                                                            constraint_vals,
                                                                            signal_constr_vals=signal_constr_vals)

        if node_num_max is not None:
            sort_hier_inds = np.argsort(self.hier_node_level[self.noninput_node_inds])
            self.influence_node_inds = list(np.asarray(self.noninput_node_inds)[sort_hier_inds][0:node_num_max])

        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self.nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self.nodes_index, constrained_inds).tolist()

        if search_main_nodes_only is False:
            if len(unconstrained_inds) <= 31:
                # If the number of nodes is less than 32, use the faster numpy-based method:
                # NOTE: 32 is a number that is hard-coded into Numpy
                M_pstates, _, _ = self.generate_state_space(unconstrained_inds)

            else:
                M_pstates = self.generate_bool_state_space(unconstrained_inds)

        else:
            if len(self.main_nodes):
                if node_num_max is None:
                    M_pstates = self.generate_bool_state_space(self.main_nodes)
                elif len(self.main_nodes) < node_num_max:
                    M_pstates, _, _ = self.generate_state_space(self.main_nodes)
                else:
                    M_pstates = self.generate_bool_state_space(self.influence_node_inds)

            else:
                raise Exception("No main nodes; cannot perform state search with "
                                "search_main_nodes_only=True.")

        sol_Mo = []
        sol_char = []

        for cvecto in M_pstates: # for each test vector:
            # Need to modify the cvect vector to hold the value of the input nodes:
            if constrained_inds is not None and constrained_vals is not None:
                cvecto[constrained_inds] = constrained_vals

            # get values for the genes we're solving for:
            sol_i, char_i = self.net_state_compute(cvecto,
                                                   A_bool_f,
                                                   n_max_steps=n_max_steps,
                                                   verbose=False,
                                                   constraint_inds = constrained_inds,
                                                   constraint_vals = constrained_vals
                                                   )

            sol_Mo.append(sol_i)
            sol_char.append(char_i)

            # if i == 0:
            #     sol_Mo.append(sol_i)
            #     sol_char.append(char_i)
            #     i += 1
            #
            # else:
            #     if sol_i not in np.asarray(sol_Mo):
            #         sol_Mo.append(sol_i)
            #         sol_char.append(char_i)

            if verbose:
                print(cvecto, sol_i, char_i)


        _, unique_inds = np.unique(sol_Mo, axis=0, return_index=True)

        sol_M = (np.asarray(sol_Mo)[unique_inds]).T
        sol_char = np.asarray(sol_char)[unique_inds]

        return sol_M, sol_char

    def generate_state_space(self,
                             c_inds: list,
                             ) -> tuple[ndarray, list, ndarray]:
        '''
        Generate a discrete state space over the range of probabilities of
        each individual gene in the network.
        '''
        c_lins = []

        for i in c_inds:
            c_lins.append(np.asarray([0, 1], dtype=int))

        cGrid = np.meshgrid(*c_lins)

        N_pts = len(cGrid[0].ravel())

        cM = np.zeros((N_pts, self.N_nodes), dtype=int)

        for i, cGrid in zip(c_inds, cGrid):
            cM[:, i] = cGrid.ravel()

        return cM, c_lins, cGrid

    def generate_bool_state_space(self,
                             c_inds: list,
                             ) -> ndarray:
        '''
        Generate a discrete state space over the range of probabilities of
        each individual gene in the network.
        '''

        # FIXME: this should not be expanded but code should be redeveloped to use only the iterator...
        c_lins = list(itertools.product([0,1], repeat=len(c_inds)))
        cM = np.zeros((len(c_lins), self.N_nodes), dtype=int)
        cM[:, c_inds] = c_lins

        return cM

    def _handle_constrained_nodes(self,
                                  constr_inds: list | None,
                                  constr_vals: list | None,
                                  signal_constr_vals: list | None = None
                                  ) -> tuple[list, list]:
        '''
        Networks will often have nodes without regulation that need to
        be constrained during optimization. This helper-method augments
        these naturally-occuring nodes with any additional constraints
        supplied by the user.
        '''
        len_constr = len(self.input_node_inds)

        if signal_constr_vals is None: # default to zero
            sig_vals = (np.int8(np.zeros(len_constr))).tolist()
        else:
            sig_vals = signal_constr_vals

        if len_constr != 0:
            if constr_inds is None or constr_vals is None:
                constrained_inds = self.input_node_inds.copy()
                constrained_vals = sig_vals
            else:
                constrained_inds = constr_inds + self.input_node_inds.copy()
                constrained_vals = constr_vals + sig_vals
        else:
            if constr_inds is None or constr_vals is None:
                constrained_inds = []
                constrained_vals = []

            else:
                constrained_inds = constr_inds*1
                constrained_vals = constr_vals*1

        return constrained_inds, constrained_vals


    #---State Space Search -----------------
    def bool_state_space(self,
                         A_bool_f: Callable,
                         constraint_inds: list | None = None,
                         constraint_vals: list | None = None,
                         signal_constr_vals: list | None = None,
                         search_main_nodes_only: bool = False,
                         n_max_steps: int = 20,
                         node_num_max: int|None = None
                         ):
        '''

        '''

        constrained_inds, constrained_vals = self._handle_constrained_nodes(constraint_inds,
                                                                            constraint_vals,
                                                                            signal_constr_vals=signal_constr_vals)

        sort_hier_inds = np.argsort(self.hier_node_level[self.noninput_node_inds])
        self.influence_node_inds = list(np.asarray(self.noninput_node_inds)[sort_hier_inds][0:node_num_max])

        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self.nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self.nodes_index, constrained_inds).tolist()

        if search_main_nodes_only is False:
            if len(unconstrained_inds) < node_num_max:
                # If the number of nodes is less than 32, use the faster numpy-based method:
                # NOTE: 32 is a number that is hard-coded into Numpy
                M_pstates, _, _ = self.generate_state_space(unconstrained_inds)

            elif node_num_max is None:
                # if it's greater than 32, numpy can't work with this, therefore use python itertools method:
                M_pstates = self.generate_bool_state_space(unconstrained_inds)

            else:
                M_pstates = self.generate_bool_state_space(self.influence_node_inds)

        else:
            if len(self.main_nodes):
                if len(self.main_nodes) < node_num_max:
                    M_pstates, _, _ = self.generate_state_space(self.main_nodes)
                elif node_num_max is None:
                    M_pstates = self.generate_bool_state_space(self.main_nodes)
                else:
                    M_pstates = self.generate_bool_state_space(self.influence_node_inds)

            else:
                raise Exception("No main nodes; cannot perform state search with "
                                "search_main_nodes_only=True.")

        net_edges = set()  # store the edges of the boolean state diagram
        pos={} # Holds node state position on the state transition diagram

        # FIXME: this should be an enumeration not a matrix of 1 gillion states in M_pstates!!
        for ci, cvecto in enumerate(M_pstates): # for each test vector:
            # Need to modify the cvect vector to hold the value of the input nodes:
            if constrained_inds is not None and constrained_vals is not None:
                cvecto[constrained_inds] = constrained_vals

            cc_i = cvecto  # initialize the function values (node values)

            for i in range(n_max_steps):
                cc_o = cc_i # save the initial value
                cc_i = np.sign(A_bool_f(cc_i)[0])  # calculate new state values

                # Need to modify the new concentrations vector to hold the value of the input nodes:
                if constrained_inds is not None and constrained_vals is not None:
                    cc_i[constrained_inds] = constrained_vals

                nde1 = str(tuple(cc_o[self.noninput_node_inds]))
                nde2 = str(tuple(cc_i[self.noninput_node_inds]))

                # nde1 = ''.join(str(int(i)) for i in cc_o[self.noninput_node_inds])
                # nde2 = ''.join(str(int(i)) for i in cc_i[self.noninput_node_inds])

                net_edges.add((nde1, nde2))
                pos[nde1] = tuple(cc_o[self.noninput_node_inds])
                pos[nde2] = tuple(cc_i[self.noninput_node_inds])

                # Detect whether we're at a steady-state:
                if (cc_i == cc_o).all() is NumpyTrue:
                    break

        boolG = nx.DiGraph(net_edges)
        return boolG, pos


    # ----Plots and Data Export---------------

    # FIXME: work this up
    def save_model_equations(self,
                             save_eqn_image: str,
                             save_eqn_csv: str | None = None,
                             ):
        '''
        Save images of the model equations, as well as a csv file that has
        model equations written in LaTeX format.

        Parameters
        -----------
        save_eqn_image : str
            The path and filename to save the main model equations as an image.

        save_reduced_eqn_image : str|None = None
            The path and filename to save the reduced main model equations as an image (if model is reduced).

        save_eqn_csv : str|None = None
            The path and filename to save the main and reduced model equations as LaTex in a csv file.

        '''
        if self._A_bool_s is None:
            raise Exception("No model built; cannot save model equations.")

        _c_vect_s = self._c_vect_s

        c_name = sp.Matrix([ci for ci in _c_vect_s])
        # eqn_net = sp.Eq(c_name, self._A_bool_s)

        for ii, (cnme, beqn) in enumerate(zip(c_name, self._A_bool_s)):
            eqn_net = sp.Eq(cnme, beqn)

            save_eqn_image_i = save_eqn_image + f'_{cnme}_.png'

            sp.preview(eqn_net,
                       viewer='file',
                       filename=save_eqn_image_i,
                       euler=False,
                       dvioptions=["-T",
                                   "tight",
                                   "-z", "0",
                                   "--truecolor",
                                   "-D 600",
                                   "-bg",
                                   "Transparent"])

        # Save the equations for the graph to a file:
        header = ['Concentrations', 'Formula']
        eqns_to_write = [[sp.latex(_c_vect_s), sp.latex(self._A_bool_s)]]

        if save_eqn_csv is not None:
            with open(save_eqn_csv, 'w', newline="") as file:
                csvwriter = csv.writer(file)  # 2. create a csvwriter object
                csvwriter.writerow(header)  # 4. write the header
                csvwriter.writerows(eqns_to_write)  # 5. write the rest of the data
