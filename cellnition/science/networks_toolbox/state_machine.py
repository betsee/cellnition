#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds and plots a state transition diagram from a solution
set and corresponding GeneNetworkModel.
'''

import os
import itertools
import numpy as np
import networkx as nx
import pygraphviz as pgv
from cellnition.science.network_models.probability_networks import (
    ProbabilityNet)
from cellnition.science.network_models.network_enums import EquilibriumType
from cellnition._util.path.utilpathmake import FileRelative
from cellnition._util.path.utilpathself import get_data_png_glyph_stability_dir
from collections import OrderedDict
from numpy import ndarray
from matplotlib import colors
from matplotlib import colormaps
from networkx import MultiDiGraph
from pygraphviz import AGraph

class StateMachine(object):
    '''
    Build and plots a state transition diagram from a solution set and
    corresponding GeneNetworkModel. This class uses time simulation,
    starting the system off at the zero vector plus every stable state in
    a supplied matrix, and by temporarily triggering signal nodes in the
    network, it then looks to see if there is a new stable state for the
    system after the perturbation. The transitions between states are
    recorded in a state transition diagram, which is allowed to have parallel
    edges. Due to complexity, self-loops are omitted.

    Public Attributes
    -----------------
    G_states : MultiDiGraph
        State transition network, showing how each steady-state of the
        network is reached through a signal transition. This is MultiDiGraph,
        which means parallel edges (meaning it is possible for different signals to
        transition the system between the same two states). For simplicity, self-loops
        are omitted from the diagrams.

    Private Attributes
    ------------------
    _gmod : GeneNetworkModel
        An instance of GeneNetworkModel

    _solsM : ndarray
        A set of steady-state solutions from _gmod.
    '''

    def __init__(self, pnet: ProbabilityNet):
        '''
        Initialize the StateMachine.

        Parameters
        ----------
        pnet : NetworkABC
            An instance of NetworkABC with an analytical model built.

        solsM : ndarray
            A set of unique steady-state solutions from the GeneNetworkModel.
            These will be the states of the StateMachine.
        '''

        self._pnet = pnet
        self.G_states = None # The state transition network

        # Path to load image assets:
        GLYPH_DIR = get_data_png_glyph_stability_dir()
        attractor_fname = FileRelative(GLYPH_DIR, 'glyph_attractor.png')
        limitcycle_fname = FileRelative(GLYPH_DIR, 'glyph_limit_cycle.png')
        saddle_fname = FileRelative(GLYPH_DIR, 'glyph_saddle.png')
        attractor_limitcycle_fname = FileRelative(GLYPH_DIR, 'glyph_attractor_limit_cycle.png')
        repellor_limitcycle_fname = FileRelative(GLYPH_DIR, 'glyph_repellor_limit_cycle.png')
        repellor_fname = FileRelative(GLYPH_DIR, 'glyph_repellor.png')
        unknown_fname = FileRelative(GLYPH_DIR, 'glyph_unknown.png')

        # Associate each equilibrium type with an image file
        self._node_image_dict = {
            EquilibriumType.attractor.name: str(attractor_fname),
            EquilibriumType.limit_cycle.name: str(limitcycle_fname),
            EquilibriumType.saddle.name: str(saddle_fname),
            EquilibriumType.attractor_limit_cycle.name: str(attractor_limitcycle_fname),
            EquilibriumType.repellor_limit_cycle.name: str(repellor_limitcycle_fname),
            EquilibriumType.repellor.name: str(repellor_fname),
            EquilibriumType.undetermined.name: str(unknown_fname)
        }

    def steady_state_solutions_search(self,
                                      beta_base: float | list = 0.25,
                                      n_base: float | list = 15.0,
                                      d_base: float | list = 1.0,
                                      verbose: bool=True,
                                      return_saddles: bool=True,
                                      N_space: int=3,
                                      search_tol: float=1.0e-15,
                                      sol_tol: float=1.0e-2,
                                      N_round_sol: int=1,
                                      order_states: bool = True
                                      ):
        '''
        Search through all possible combinations of signal node values
        and collect and identify all equilibrium points of the system.

        '''
        # FIXME: do we want this to have additional node constraints?

        sig_lin = [1.0e-6, 1.0]
        sig_lin_set = [sig_lin for i in self._pnet.input_node_inds]

        sigGrid = np.meshgrid(*sig_lin_set)

        N_vocab = len(sigGrid[0].ravel())

        sig_test_set = np.zeros((N_vocab, len(self._pnet.input_node_inds)))

        for i, sigM in enumerate(sigGrid):
            sig_test_set[:, i] = sigM.ravel()

        solsM_allo = []
        charM_allo = []
        sols_list = []

        for sigis in sig_test_set:
            print(f'Signals: {np.round(sigis, 1)}')
            solsM, sol_M_char, sol_0 = self._pnet.solve_probability_equms(constraint_inds=None,
                                                                    constraint_vals=None,
                                                                    signal_constr_vals=sigis.tolist(),
                                                                    d_base=d_base,
                                                                    n_base=n_base,
                                                                    beta_base=beta_base,
                                                                    N_space=N_space,
                                                                    search_tol=search_tol,
                                                                    sol_tol=sol_tol,
                                                                    N_round_sol=N_round_sol,
                                                                    verbose=verbose,
                                                                    return_saddles=return_saddles
                                                                    )


            solsM_allo.append(solsM)  # append all unique sols
            charM_allo.append(sol_M_char)  # append the sol stability characterization tags
            sols_list.append(solsM)
            if verbose:
                print('----')

        # Perform a merger of sols into one array and find only the unique solutions
        solsM_all = np.zeros((self._pnet.N_nodes, 1))  # include the zero state
        charM_all = [EquilibriumType.undetermined.name]  # set the zero state to undetermined by default

        for i, (soli, chari) in enumerate(zip(solsM_allo, charM_allo)):
            solsM_all = np.hstack((solsM_all, soli))
            charM_all.extend(chari)

            # _, inds_solsM_all_unique = np.unique(np.round(solsM_all, 1), axis=1, return_index=True)
        _, inds_solsM_all_unique = np.unique(np.round(solsM_all, 1)[self._pnet.noninput_node_inds, :], axis=1,
                                             return_index=True)

        solsM_all = solsM_all[:, inds_solsM_all_unique]
        charM_all = np.asarray(charM_all)[inds_solsM_all_unique]

        if np.zeros(self._pnet.N_nodes).T in np.round(solsM_all, 1).T:
            zer_index = np.round(solsM_all, 1).T.tolist().index(np.zeros(self._pnet.N_nodes).T.tolist())
            if charM_all[
                zer_index] == EquilibriumType.undetermined.name:  # if it hasn't been alogrithmically set already...
                charM_all.T[zer_index] = EquilibriumType.attractor.name  # update the state to an attractor

        else:
            charM_all[0] = EquilibriumType.saddle.name  # update the state to a saddle node

        if order_states: # order states as distance from the zero vector:
            solsM_all, charM_all = self._order_states_by_distance(solsM_all, charM_all)

        # set of all states referencing only the hub nodes; rounded to one decimal:
        state_set = np.round(solsM_all[self._pnet.noninput_node_inds, :].T, 1).tolist()

        states_dict = OrderedDict()
        for sigi in sig_test_set:
            states_dict[tuple(sigi)] = {'States': [], 'Stability': []}

        for sigi, state_subseto in zip(sig_test_set, sols_list):
            state_subset = state_subseto[self._pnet.noninput_node_inds, :]
            for target_state in np.round(state_subset, 1).T.tolist():
                if target_state in state_set:
                    state_match_index = state_set.index(target_state)
                    states_dict[tuple(sigi)]['States'].append(state_match_index)
                    states_dict[tuple(sigi)]['Stability'].append(charM_all[state_match_index])
                else:
                    print('match not found!')
                    states_dict[tuple(sigi)]['States'].append(np.nan)
                    states_dict[tuple(sigi)]['Stability'].append(np.nan)

        return solsM_all, charM_all, sols_list, states_dict, sig_test_set

    def create_transition_network(self,
                                  states_dict: dict,
                                  sig_test_set: list|ndarray,
                                  solsM_all: ndarray|list,
                                  dt: float = 1.0e-3,
                                  tend: float = 100.0,
                                  space_sig: float = 30.0,
                                  delta_sig: float = 30.0,
                                  t_relax: float = 15.0,
                                  dt_samp: float=0.15,
                                  verbose: bool = True,
                                  match_tol: float = 0.05,
                                  d_base: float = 1.0,
                                  n_base: float = 15.0,
                                  beta_base: float = 0.25
                                  ) -> tuple[set, MultiDiGraph]:
        '''
        Build a state transition matrix/diagram by starting the system
        in different states and seeing which state it ends up in after
        a time simulation. This method iterates through all 'signal'
        nodes of a network and sets them to the sigmax level, harvesting
        the new steady-state reached after perturbing the network.

        Parameters
        ----------
        dt: float = 1.0e-3
            Timestep for the time simulation.

        tend: float = 100.0
            End time for the time simulation. This must be long
            enough to allow the system to reach the second steady-state.

        sig_tstart: float = 33.0
            The time to start the signal perturbation. Care must be taken
            to ensure enough time is allotted prior to starting the perturbation
            for the system to have reached an initial steady state.

        sig_tend: float = 66.0
            The time to end the signal perturbation.

        sig_base: float = 1.0
            Baseline magnitude of the signal node.

        sig_active: float = 1.0
            Magnitude of the signal pulse during the perturbation.

        delta_window: float = 1.0
            Time to sample prior to the application of the signal perturbation,
            in which the initial steady-state is collected.

        verbose: bool = True
            Print out log statements (True)?

        tol: float = 1.0e-6
            Tolerance, below which a state is accepted as a match. If the state
            match error is above tol, it is added to the matrix as a new state.

        '''

        # States for perturbation of the zero state inputs
        # Let's start the system off in the zero vector, then
        # temporarily perturb the system with each signal set and see what the final state is after
        # the perturbation.

        sig_inds = self._pnet.input_node_inds
        N_sigs = len(sig_inds)

        # We want all signals on at the same time (we want the sim to end before
        # the signal changes again:
        sig_times = [(space_sig, delta_sig + space_sig) for i in range(N_sigs)]

        transition_edges_set = set()

        num_step = 0

        # We first want to step through all 'held' signals and potentially multistable states:
        for sig_base_set, sc_dict in states_dict.items():

            states_set = sc_dict['States']

            # Get an integer label for the 'bitstring' of signal node inds defining the base:
            base_input_label = self._get_integer_label(sig_base_set)

            # We then step through all possible perturbation signals:
            for sig_val_set in sig_test_set:

                # Get an integer label for the 'bitstring' of signal node inds on perturbation:
                pert_input_label = self._get_integer_label(sig_val_set)

                # we want the signals to go from zero to the new held state defined in sig_val set:
                sig_mags = [(sigb, sigi) for sigb, sigi in zip(sig_base_set, sig_val_set)]

                # We want to use each state in states_set as the initial condition:
                for si in states_set:
                    num_step += 1

                    cvecti = 1 * solsM_all[:, si]

                    ctime, tvect = self._pnet.run_time_sim(tend, dt, cvecti.copy(),
                                                     sig_inds=sig_inds,
                                                     sig_times=sig_times,
                                                     sig_mag=sig_mags,
                                                     dt_samp=dt_samp,
                                                     constrained_inds=None,
                                                     constrained_vals=None,
                                                     d_base=d_base,
                                                     n_base=n_base,
                                                     beta_base=beta_base
                                                     )

                    # Create sampling windows in time: FIXME: do only once!
                    window1 = (0.0 + t_relax, sig_times[0][0])
                    window2 = (sig_times[0][0] + t_relax, sig_times[0][1])
                    window3 = (sig_times[0][1] + t_relax, tend)
                    # get the indices for each window time:
                    inds_win1 = (
                    self._get_index_from_val(tvect, window1[0], dt_samp),
                    self._get_index_from_val(tvect, window1[1], dt_samp))
                    inds_win2 = (
                    self._get_index_from_val(tvect, window2[0], dt_samp),
                    self._get_index_from_val(tvect, window2[1], dt_samp))
                    inds_win3 = (
                    self._get_index_from_val(tvect, window3[0], dt_samp),
                    self._get_index_from_val(tvect, window3[1], dt_samp))

                    c_initial = np.mean(ctime[inds_win1[0]:inds_win1[1], :], axis=0)
                    # var_c_initial = np.sum(np.std(ctime[inds_win1[0]:inds_win1[1], :], axis=0))
                    initial_state, match_error_initial = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                          c_initial[self._pnet.noninput_node_inds])

                    if match_error_initial > match_tol: # if state is unmatched, flag it with a nan
                        initial_state = np.nan

                    c_held = np.mean(ctime[inds_win2[0]:inds_win2[1], :], axis=0)
                    # var_c_held = np.sum(np.std(ctime[inds_win2[0]:inds_win2[1], :], axis=0))
                    held_state, match_error_held = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                    c_held[self._pnet.noninput_node_inds])

                    if match_error_held > match_tol: # if state is unmatched, flag it
                        held_state = np.nan

                    c_final = np.mean(ctime[inds_win3[0]:inds_win3[1], :], axis=0)
                    # var_c_final = np.sum(np.std(ctime[inds_win3[0]:inds_win3[1], :], axis=0))
                    final_state, match_error_final = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                      c_final[self._pnet.noninput_node_inds])

                    if match_error_final > match_tol: # if state is unmatched, flag it
                        final_state = np.nan

                    if verbose:
                        print(num_step)
                        print(f'Transition State {initial_state} to {held_state} via {pert_input_label}')
                        print(f'Transition State {held_state} to {final_state} via {base_input_label}')
                        print('------')

                    transition_edges_set.add((initial_state, held_state, pert_input_label))
                    transition_edges_set.add((held_state, final_state, base_input_label))

        # The first thing we do after the construction of the
        # transition edges set is make a multidigraph and
        # use networkx to pre-process & simplify it, removing inaccessible states
        # (states with no non-self input degree)

        # Create the multidigraph:
        GG = nx.MultiDiGraph()

        for ndei, ndej, trans_label_ij in list(transition_edges_set):
            GG.add_edge(ndei, ndej, key=f'I{trans_label_ij}')

        # Remove nodes that have no input degree other than their own self-loop:
        nodes_with_selfloops = list(nx.nodes_with_selfloops(GG))
        for node_lab, node_in_deg in list(GG.in_degree()):
            if (node_in_deg == 1 and node_lab in nodes_with_selfloops) or node_in_deg == 0:
                GG.remove_node(node_lab)

        return transition_edges_set, GG

    def plot_state_transition_network(self,
                                      nodes_list: list,
                                      edges_list: list,
                                      charM_all: list|ndarray,
                                      save_file: str|None = None,
                                      graph_layout: str='dot'
                                      ):
        '''

        '''
        # FIXME: we probably also want the option to just plot a subset of the state dict?
        # FIXME: Should these be options in the method?
        img_pos = 'bc'  # position of the glyph in the node
        subcluster_font = 'DejaVu Sans Bold'
        node_shape = 'ellipse'
        clr_map = 'rainbow_r'
        nde_font_color = 'Black'
        hex_transparency = '80'

        # Try to make a nested graph:
        G = pgv.AGraph(strict=False,
                       fontname=subcluster_font,
                       splines=True,
                       directed=True,
                       concentrate=False,
                       # compound=True,
                       dpi=300)

        cmap = colormaps[clr_map]
        norm = colors.Normalize(vmin=0, vmax=len(nodes_list))
        # Add all the nodes:
        for nde_i in nodes_list:
            nde_index = nodes_list.index(nde_i)
            nde_color = colors.rgb2hex(cmap(norm(nde_index)))
            nde_color += hex_transparency  # add some transparancy to the node

            char_i = charM_all[nde_i] # Get the stability characterization for this state

            G.add_node(nde_i,
                           label=f'State {nde_i}',
                           labelloc='t',
                           image=self._node_image_dict[char_i],
                           imagepos=img_pos,
                           shape=node_shape,
                           fontcolor=nde_font_color,
                           style='filled',
                           fillcolor=nde_color)


        # Add all the edges:
        for nde_i, nde_j, trans_ij in edges_list:
            G.add_edge(nde_i, nde_j, label=trans_ij)

        if save_file is not None:
            G.layout(prog=graph_layout)
            G.draw(save_file)

        return G


    def _order_states_by_distance(self, solsM_all, charM_all):
        '''
        Re-arrange the supplied solution matrix so that the states are
        progressively closer to one another, in order to see a more
        logical transition through the network with perturbation.
        '''
        zer_sol = solsM_all[:, 0]
        dist_list = []

        for soli in solsM_all.T:
            # calculate the "distance" between the two solutions
            # and append to the distance list:
            dist_list.append(np.sqrt(np.sum((zer_sol[self._pnet.noninput_node_inds] -
                                             soli[self._pnet.noninput_node_inds]) ** 2)))

        inds_sort = np.argsort(dist_list)

        solsM_all = solsM_all[:, inds_sort]
        charM_all = charM_all[inds_sort]

        return solsM_all, charM_all

    def _get_index_from_val(self, val_vect: ndarray, val: float, val_overlap: float):
        '''
        Given a value in an array, this method returns the index
        of the closest value in the array.

        Parameters
        -----------
        val_vect : ndarray
            The vector of values to which the closest index to val is sought.

        val: float
            A value for which the closest matched index in val_vect is to be
            returned.

        val_overlap: float
            An amount of overlap to include in search windows to ensure the
            search will return at least one index.
        '''
        inds_l = (val_vect <= val + val_overlap).nonzero()[0]
        inds_h = (val_vect >= val - val_overlap).nonzero()[0]
        indo = np.intersect1d(inds_l, inds_h)
        if len(indo):
            ind = indo[0]
        else:
            raise Exception("No matching index was found.")

        return ind

    def _get_integer_label(self, sig_set: tuple|list|ndarray) -> int:
        '''
        Given a list of digits representing a bit string
        (i.e. a list of values close to zero or 1), this method
        treats the list as a binary bit-string and returns the
        base-2 integer representation of the bit-string.

        Parameters
        ----------
        sig_set : list[float|int]
            The list containing floats or ints that are taken to represent
            a bit string.

        Returns
        -------
        An integer representation of the binary bit-string.

        '''
        base_str = ''
        for sigi in sig_set:
            base_str += str(int(sigi))
        return int(base_str, 2)

    def get_state_distance_matrix(self, solsM_all):
        '''
        Returns a matrix representing the L2 norm 'distance'
        between each state in the array of all possible states.

        '''
        num_sols = solsM_all.shape[1]
        state_distance_M = np.zeros((num_sols, num_sols))
        for i in range(num_sols):
            for j in range(num_sols):
                # d_states = np.sqrt(np.sum((solsM_all[:,i] - solsM_all[:, j])**2))
                d_states = np.sqrt(
                    np.sum((solsM_all[self._pnet.noninput_node_inds, i] -
                            solsM_all[self._pnet.noninput_node_inds, j]) ** 2))
                state_distance_M[i, j] = d_states

        return state_distance_M

    def _find_state_match(self,
                         solsM: ndarray,
                         cvecti: list | ndarray) -> tuple:
        '''
        Given a matrix of possible states and a concentration vector,
        return the state that best-matches the concentration vector,
        along with an error for the comparison.

        Parameters
        ----------
        solsM : ndarray
            A matrix with a set of steady-state solutions arranged in
            columns.

        cvecti : list
            A list of concentrations with which to compare with each
            steady-state in solsM, in order to select a best-match state
            from solsM to cvecti.

        Returns
        -------
        state_best_match
            The index of the best-match state in solsM
        err
            The error to the match
        '''

        # now what we need is a pattern match from concentrations to the stable states:
        errM = []
        for soli in solsM.T:
            sdiff = soli - cvecti
            errM.append(np.sum(sdiff ** 2))
        errM = np.asarray(errM)
        state_best_match = (errM == errM.min()).nonzero()[0][0]

        return state_best_match, errM[state_best_match]



