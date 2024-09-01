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
import copy
import numpy as np
from scipy.cluster.hierarchy import fclusterdata
import networkx as nx
import pygraphviz as pgv
from cellnition.science.network_models.probability_networks import (
    ProbabilityNet)
from cellnition.science.network_models.network_enums import EquilibriumType
from cellnition._util.path.utilpathmake import FileRelative
from cellnition._util.path.utilpathself import get_data_png_glyph_stability_dir
from collections import OrderedDict
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
        hidden_fname = FileRelative(GLYPH_DIR, 'glyph_hidden.png')

        # Associate each equilibrium type with an image file
        self._node_image_dict = {
            EquilibriumType.attractor.name: str(attractor_fname),
            EquilibriumType.limit_cycle.name: str(limitcycle_fname),
            EquilibriumType.saddle.name: str(saddle_fname),
            EquilibriumType.attractor_limit_cycle.name: str(attractor_limitcycle_fname),
            EquilibriumType.repellor_limit_cycle.name: str(repellor_limitcycle_fname),
            EquilibriumType.repellor.name: str(repellor_fname),
            EquilibriumType.undetermined.name: str(unknown_fname),
            EquilibriumType.hidden.name: str(hidden_fname)
        }

    def run_state_machine(self, beta_base: float | list = 0.25,
                          n_base: float | list = 15.0,
                          d_base: float | list = 1.0,
                          verbose: bool=True,
                          return_saddles: bool=True,
                          N_space: int=3,
                          search_tol: float=1.0e-15,
                          sol_tol: float=1.0e-2,
                          N_round_sol: int=1,
                          dt: float = 1.0e-3,
                          space_sig: float = 30.0,
                          delta_sig: float = 30.0,
                          t_relax: float = 15.0,
                          dt_samp: float = 0.15,
                          match_tol: float = 0.05,
                          save_graph_file: str|None =None,
                          save_transition_net_image: str | None = None,
                          save_perturbation_net_image: str|None = None,
                          graph_layout: str = 'dot',
                          remove_inaccessible_states: bool = True,
                          unique_sol_index: bool=True,
                          search_cycle_nodes_only: bool = False
                          ) -> MultiDiGraph:
        '''
        Run all steps to generate a state transition network and associated
        network image.
        '''

        # Run a search through input space to obtain all unique steady states and their equ'm characterization:
        (solsM_all,
         charM_all,
         sols_list,
         states_dict,
         sig_test_set) = self.steady_state_solutions_search(beta_base=beta_base,
                                                            n_base=n_base,
                                                            d_base=d_base,
                                                            verbose=verbose,
                                                            return_saddles=return_saddles,
                                                            N_space=N_space,
                                                            search_tol=search_tol,
                                                            sol_tol=sol_tol,
                                                            N_round_sol=N_round_sol,
                                                            search_cycle_nodes_only=search_cycle_nodes_only
                                                            )

        # save entities to the object:
        self.solsM_all = solsM_all
        self.charM_all = charM_all
        self.sols_list = sols_list
        self.states_dict = states_dict
        self.sig_test_set = sig_test_set # should be called "input space states"

        # Create the edges of the transition network:
        transition_edges_set, pert_edges_set, G_nx = self.create_transition_network(states_dict, sig_test_set, solsM_all,
                                                                     dt=dt,
                                                                     space_sig=space_sig,
                                                                     delta_sig=delta_sig,
                                                                     t_relax=t_relax,
                                                                     dt_samp=dt_samp,
                                                                     verbose=verbose,
                                                                     match_tol=match_tol,
                                                                     d_base=d_base,
                                                                     n_base=n_base,
                                                                     beta_base=beta_base,
                                                                     remove_inaccessible_states=remove_inaccessible_states,
                                                                     save_graph_file=save_graph_file,
                                                                     )

        self.transition_edges_set = transition_edges_set
        self.pert_edges_set = pert_edges_set
        self.G_nx = G_nx

        if save_perturbation_net_image:
            nodes_list = sorted(G_nx.nodes())
            # Generate a perturbation network plot
            G_pert = self.plot_state_perturbation_network(self.pert_edges_set,
                                                           self.states_dict,
                                                          solsM_all,
                                                           nodes_listo=nodes_list,
                                                           save_file=save_perturbation_net_image,
                                                          graph_layout=graph_layout,
                                                          unique_sol_index=unique_sol_index,
                                                          N_round_sol=N_round_sol)

        if save_transition_net_image is not None: # save an image of the network to file:
            # get nodes and edges list:
            nodes_list = sorted(G_nx.nodes())
            edges_list = list(G_nx.edges)

            G_gv = self.plot_state_transition_network(nodes_list,
                                                      edges_list,
                                                      solsM_all,
                                                      charM_all,
                                                      save_file=save_transition_net_image,
                                                      graph_layout=graph_layout,
                                                      use_unique_sol_index=unique_sol_index,
                                                      N_round_sol=N_round_sol
                                                      )

        return G_nx


    def steady_state_solutions_search(self,
                                      beta_base: float | list,
                                      n_base: float | list,
                                      d_base: float | list,
                                      verbose: bool=True,
                                      return_saddles: bool=True,
                                      N_space: int=3,
                                      search_tol: float=1.0e-15,
                                      sol_tol: float=1.0e-2,
                                      N_round_sol: int=1,
                                      search_cycle_nodes_only: bool = False,
                                      cluster_threshhold: float=0.1,
                                      cluster_method: str = 'distance',
                                      sig_lino: list|None = None
                                      ):
        '''
        Search through all possible combinations of signal node values
        and collect and identify all equilibrium points of the system.

        '''

        if sig_lino is None:
            sig_lin = [1.0e-6, 1.0]
        else:
            sig_lin = sig_lino

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
                                                                    return_saddles=return_saddles,
                                                                    search_cycle_nodes_only=search_cycle_nodes_only
                                                                    )




            solsM_allo.append(solsM)  # append all unique sols
            charM_allo.append(sol_M_char)  # append the sol stability characterization tags
            sols_list.append(solsM)
            if verbose:
                print('----')

        # Perform a merger of sols into one array and find only the unique solutions
        # solsM_all = np.zeros((self._pnet.N_nodes, 1))  # include the zero state
        # charM_all = [EquilibriumType.undetermined.name]  # set the zero state to undetermined by default

        solsM_all = None
        charM_all = []

        for i, (soli, chari) in enumerate(zip(solsM_allo, charM_allo)):
            if i == 0:
                solsM_all = soli
            else:
                solsM_all = np.hstack((solsM_all, soli))
            charM_all.extend(chari)

        # First append all attractor types as an integer value as a way to
        # further distinguish states by their dynamics:
        charM_all_vals = []
        for ci in charM_all:
            attr_type = getattr(EquilibriumType, ci, None)
            if attr_type is not None:
                charM_all_vals.append(attr_type.value)

        solsM_all_char = np.vstack((solsM_all, charM_all_vals))

        select_inds = []
        select_inds.extend(self._pnet.noninput_node_inds)
        select_inds.append(-1)

        # # # first use numpy unique on rounded set of solutions to exclude similar cases:
        # _, inds_solsM_all_unique = np.unique(np.round(solsM_all_char[select_inds, :],
        #                                               N_round_sol), return_index=True, axis=1)
        # #
        # solsM_all_char = solsM_all_char[:, inds_solsM_all_unique]
        # charM_all = np.asarray(charM_all)[inds_solsM_all_unique]

        # Next use a clustering algorithm to disregard solutions that are very close
        # to one another in terms of an Euclidian vector distance:

        unique_sol_clusters = fclusterdata(solsM_all_char[select_inds, :].T,
                                           t=cluster_threshhold,
                                           criterion=cluster_method)

        cluster_index = np.unique(unique_sol_clusters)

        cluster_pool = [[] for i in cluster_index]
        for i, clst_i in enumerate(unique_sol_clusters):
            cluster_pool[int(clst_i) - 1].append(i)

        solsM_all_unique = np.zeros((self._pnet.N_nodes, len(cluster_pool)))
        charM_all_unique = []

        for ii, sol_i in enumerate(cluster_pool):
            if len(sol_i):
                solsM_all_unique[self._pnet.noninput_node_inds, ii] = (
                    np.mean(solsM_all[:, sol_i][self._pnet.noninput_node_inds], 1))
                # print(np.asarray(charM_all)[sol_i])
                charM_all_unique.append(charM_all[sol_i[0]])

        # redefine the solsM_all and charM_all data structures:
        solsM_all = solsM_all_unique
        charM_all = np.asarray(charM_all_unique)

        # # # first use numpy unique on rounded set of solutions to exclude similar cases:
        _, inds_solsM_all_unique = np.unique(np.round(solsM_all[self._pnet.noninput_node_inds, :],
                                                      N_round_sol), return_index=True, axis=1)
        # #
        solsM_all = solsM_all[:, inds_solsM_all_unique]
        charM_all = charM_all[inds_solsM_all_unique]

        states_dict = OrderedDict()
        for sigi in sig_test_set:
            states_dict[tuple(sigi)] = {'States': [], 'Stability': []}

        for sigi, state_subseto in zip(sig_test_set, sols_list):
            state_subset = state_subseto[self._pnet.noninput_node_inds, :]
            for target_state in state_subset.T.tolist():
                state_match_index, err_match = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                       target_state)
                if state_match_index not in states_dict[tuple(sigi)]['States']:
                    states_dict[tuple(sigi)]['States'].append(state_match_index)
                    states_dict[tuple(sigi)]['Stability'].append(charM_all[state_match_index])

        return solsM_all, charM_all, sols_list, states_dict, sig_test_set

    def create_transition_network(self,
                                  states_dict: dict,
                                  sig_test_set: list|ndarray,
                                  solsM_allo: ndarray,
                                  dt: float = 5.0e-3,
                                  delta_sig: float = 40.0,
                                  t_relax: float = 10.0,
                                  dt_samp: float=0.1,
                                  verbose: bool = True,
                                  match_tol: float = 0.05,
                                  d_base: float|list[float] = 1.0,
                                  n_base: float|list[float] = 15.0,
                                  beta_base: float|list[float] = 0.25,
                                  remove_inaccessible_states: bool=False,
                                  save_graph_file: str|None = None,
                                  save_time_runs: bool=False
                                  ) -> tuple[set, set, MultiDiGraph]:
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

        # make a copy of solsM_all:
        solsM_all = solsM_allo.copy()

        # make a copy of the states dict that's only used for modifications:
        states_dict_2 = copy.deepcopy(states_dict)

        # States for perturbation of the zero state inputs
        # Let's start the system off in the zero vector, then
        # temporarily perturb the system with each signal set and see what the final state is after
        # the perturbation.

        sig_inds = self._pnet.input_node_inds
        N_sigs = len(sig_inds)

        # We want all signals on at the same time (we want the sim to end before
        # the signal changes again:
        sig_times = [(delta_sig, 2*delta_sig) for i in range(N_sigs)]

        tend = sig_times[-1][1] + delta_sig

        transition_edges_set = set()
        perturbation_edges_set = set()

        num_step = 0

        # Get the full time vector and the sampled time vector (tvectr)
        tvect, tvectr = self._pnet.make_time_vects(tend, dt, dt_samp)

        # Create sampling windows in time:
        window1 = (0.0 + t_relax, sig_times[0][0])
        window2 = (sig_times[0][0] + t_relax, sig_times[0][1])
        window3 = (sig_times[0][1] + t_relax, tend)
        # get the indices for each window time:
        inds_win1 = (
            self._get_index_from_val(tvectr, window1[0], dt_samp),
            self._get_index_from_val(tvectr, window1[1], dt_samp))
        inds_win2 = (
            self._get_index_from_val(tvectr, window2[0], dt_samp),
            self._get_index_from_val(tvectr, window2[1], dt_samp))
        inds_win3 = (
            self._get_index_from_val(tvectr, window3[0], dt_samp),
            self._get_index_from_val(tvectr, window3[1], dt_samp))

        _all_time_runs = []

        # We want to step through all 'held' signals and potentially multistable states:
        for base_input_label, (sig_base_set, sc_dict) in enumerate(states_dict.items()):

            states_set = sc_dict['States']

            # Get an integer label for the 'bitstring' of signal node inds defining the base:
            # base_input_label = self._get_integer_label(sig_base_set)

            # We then step through all possible perturbation signals:
            for pert_input_label, sig_val_set in enumerate(sig_test_set):

                # Get an integer label for the 'bitstring' of signal node inds on perturbation:
                # pert_input_label = self._get_integer_label(sig_val_set)

                # we want the signals to go from zero to the new held state defined in sig_val set:
                sig_mags = [(sigb, sigi) for sigb, sigi in zip(sig_base_set, sig_val_set)]

                # We want to use each state in states_set as the initial condition:
                for si in states_set:
                    # Initial state vector: add the small non-zero amount to prevent 0/0 in Hill functions:
                    cvecti = 1 * solsM_all[:, si] + self._pnet.p_min

                    # if verbose:
                    #     print(f'Start state {si}...')

                    c_signals = self._pnet.make_pulsed_signals_matrix(tvect, sig_inds, sig_times, sig_mags)

                    ctime = self._pnet.run_time_sim(tvect, tvectr, cvecti.copy(),
                                                           sig_inds=sig_inds,
                                                           sig_vals=c_signals,
                                                           constrained_inds=None,
                                                           constrained_vals=None,
                                                           d_base=d_base,
                                                           n_base=n_base,
                                                           beta_base=beta_base
                                                           )

                    c_initial = np.mean(ctime[inds_win1[0]:inds_win1[1], :], axis=0)
                    # var_c_initial = np.sum(np.std(ctime[inds_win1[0]:inds_win1[1], :], axis=0))

                    # match the network state to one that only involves the hub nodes:
                    initial_state, match_error_initial = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                          c_initial[self._pnet.noninput_node_inds])
                    # initial_state, match_error_initial = self._find_state_match(solsM_all, c_initial)

                    if match_error_initial > match_tol: # if state is unmatched, flag it with a nan
                        if verbose:
                            print(f'Initial state not found; adding new state to the solution set...')
                        solsM_all = np.column_stack((solsM_all, c_initial))
                        initial_state = solsM_all.shape[1] - 1

                        # Update the states listing for this input state set
                        sc_dict2 = states_dict_2[sig_base_set]['States']
                        sc_dict2.append(initial_state)
                        states_dict_2[sig_base_set]['States'] = sc_dict2


                    c_held = np.mean(ctime[inds_win2[0]:inds_win2[1], :], axis=0)
                    # var_c_held = np.sum(np.std(ctime[inds_win2[0]:inds_win2[1], :], axis=0))

                    held_state, match_error_held = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                    c_held[self._pnet.noninput_node_inds])
                    # held_state, match_error_held = self._find_state_match(solsM_all, c_held)

                    if match_error_held > match_tol: # if state is unmatched, flag it
                        if verbose:
                            print(f'Held state not found; adding new state to the solution set...')
                        solsM_all = np.column_stack((solsM_all, c_held))
                        held_state = solsM_all.shape[1] -1

                        # Update the states listing for this input state set
                        sc_dict2 = states_dict_2[sig_base_set]['States']
                        sc_dict2.append(held_state)
                        states_dict_2[sig_base_set]['States'] = sc_dict2

                    c_final = np.mean(ctime[inds_win3[0]:inds_win3[1], :], axis=0)
                    # var_c_final = np.sum(np.std(ctime[inds_win3[0]:inds_win3[1], :], axis=0))
                    final_state, match_error_final = self._find_state_match(solsM_all[self._pnet.noninput_node_inds, :],
                                                                      c_final[self._pnet.noninput_node_inds])
                    # final_state, match_error_final = self._find_state_match(solsM_all, c_final)

                    if match_error_final > match_tol: # if state is unmatched, add it to the system
                        if verbose:
                            print(f'Final state not found; adding new state to the solution set...')
                        solsM_all = np.column_stack((solsM_all, c_final))
                        final_state = solsM_all.shape[1] -1

                        # Update the states listing for this input state set
                        sc_dict2 = states_dict_2[sig_base_set]['States']
                        sc_dict2.append(final_state)
                        states_dict_2[sig_base_set]['States'] = sc_dict2

                    if verbose:
                        print(num_step)

                    if initial_state is not np.nan and held_state is not np.nan and final_state is not np.nan:
                        transition_edges_set.add((initial_state, held_state, pert_input_label))
                        transition_edges_set.add((held_state, final_state, base_input_label))

                        if verbose:
                            print(f'Transition State {initial_state} to {held_state} via {pert_input_label}')
                            print(f'Transition State {held_state} to {final_state} via {base_input_label}')

                        if initial_state != final_state:  # add this to the perturbed transitions:
                            perturbation_edges_set.add((initial_state, final_state, pert_input_label, base_input_label))

                            if verbose:
                                print(f'Perturbed Transition from State {initial_state} to {final_state} via '
                                      f'{pert_input_label}')

                    else:
                        if verbose:
                            print(f'Warning: {initial_state} to {held_state} via {pert_input_label} not added \n '
                                  f'as state match not found! \n'
                                  f'Match errors {match_error_initial, match_error_held, match_error_final}')

                    _all_time_runs.append(ctime.copy())
                    num_step += 1

                    if verbose:
                        # print(f'Match errors {match_error_initial, match_error_held, match_error_final}')
                        print('------')

        # The first thing we do after the construction of the
        # transition edges set is make a multidigraph and
        # use networkx to pre-process & simplify it, removing inaccessible states
        # (states with no non-self input degree)

        if save_time_runs:
            self._all_time_runs = _all_time_runs
        else:
            self._all_time_runs = None

        self._solsM_all = solsM_all
        self._states_dict = states_dict_2

        # Create the multidigraph:
        GG = nx.MultiDiGraph()

        for ndei, ndej, trans_label_ij in list(transition_edges_set):
            # Annoyingly, nodes must be strings in order to save properly...
            GG.add_edge(str(ndei), str(ndej), key=f'I{trans_label_ij}')

        if remove_inaccessible_states:
            # Remove nodes that have no input degree other than their own self-loop:
            nodes_with_selfloops = list(nx.nodes_with_selfloops(GG))
            for node_lab, node_in_deg in list(GG.in_degree()):
                if (node_in_deg == 1 and node_lab in nodes_with_selfloops) or node_in_deg == 0:
                    GG.remove_node(node_lab)

        if save_graph_file:
            nx.write_gml(GG, save_graph_file)

        return transition_edges_set, perturbation_edges_set, GG

    def sim_time_trajectory(self,
                            starting_state_i: int,
                            solsM_all: ndarray,
                            input_list: list[str],
                            sig_test_set: list|ndarray,
                            dt: float=1.0e-3,
                            dt_samp: float=0.1,
                            input_hold_duration: float = 30.0,
                            t_wait: float = 10.0,
                            verbose: bool = True,
                            match_tol: float = 0.05,
                            d_base: float|list[float] = 1.0,
                            n_base: float|list[float] = 15.0,
                            beta_base: float|list[float] = 0.25,
                            time_wobble: float = 0.0,
                            ):
        '''
        Use a provided starting state and a list of input signals to hold for
        a specified duration to simulate a time trajectory of the state machine.

        Parameters
        ----------

        Returns
        -------
        '''
        c_vecti = solsM_all[:, starting_state_i]  # get the starting state concentrations

        sig_inds = self._pnet.input_node_inds

        N_phases = len(input_list)
        end_t = N_phases * input_hold_duration

        time_noise = np.random.uniform(0.0, time_wobble)

        phase_time_tuples = [(i * input_hold_duration, (i + 1) * input_hold_duration + time_noise) for i in range(N_phases)]

        # Get the full time vector and the sampled time vector (tvectr)
        tvect, tvectr = self._pnet.make_time_vects(end_t, dt, dt_samp)

        # list of tuples with indices defining start and stop of phase averaging region (for state matching solutions)
        c_ave_phase_inds = []
        for ts, te in phase_time_tuples:
            rtinds = self._pnet.get_interval_inds(tvectr, ts, te, t_wait=t_wait)
            c_ave_phase_inds.append((rtinds[0], rtinds[-1]))

        # Get the dictionary that allows us to convert between input signal labels and actual held signal values:
        signal_lookup_dict = self._get_input_signals_from_label_dict(sig_test_set)

        # Generate a signals matrix:
        sig_M = np.zeros((len(tvect), self._pnet.N_nodes))

        for sig_label, (ts, te) in zip(input_list, phase_time_tuples):
            # Get the indices for the time this phase is active:
            tinds_phase = self._pnet.get_interval_inds(tvect, ts, te, t_wait=0.0)

            sig_vals = signal_lookup_dict[sig_label]

            for si, sigv in zip(sig_inds, sig_vals):
                sig_M[tinds_phase, si] = sigv

        # now we're ready to run the time sim:
        ctime = self._pnet.run_time_sim(tvect, tvectr, c_vecti.copy(),
                                        sig_inds=sig_inds,
                                        sig_vals=sig_M,
                                        constrained_inds=None,
                                        constrained_vals=None,
                                        d_base=d_base,
                                        n_base=n_base,
                                        beta_base=beta_base
                                        )

        # now we want to state match based on average concentrations in each held-input phase:
        matched_states = []
        for i, (si, ei) in enumerate(c_ave_phase_inds):
            c_ave = np.mean(ctime[si:ei, :], axis=0)
            state_matcho, match_error = self._find_state_match(solsM_all, c_ave)
            if match_error < match_tol:
                state_match = state_matcho

                matched_states.append(state_match)
                if verbose:
                    print(f'Phase {i} state matched to State {state_match} with input {input_list[i]}')
            else:
                matched_states.append(np.nan)
                if verbose:
                    print(f'Warning! Phase {i} state matched not found (match error: {match_error})!')

        return tvectr, ctime, matched_states, c_ave_phase_inds

    def plot_state_transition_network(self,
                                      nodes_listo: list,
                                      edges_list: list,
                                      charM_all: list|ndarray,
                                      save_file: str|None = None,
                                      graph_layout: str='dot',
                                      mono_edge: bool = False,
                                      rank: str='same',
                                      constraint: bool = False,
                                      concentrate: bool = True,
                                      fontsize: float = 18.0,
                                      node_colors: list|None = None
                                      ):
        '''

        '''
        # FIXME: we probably also want the option to just plot a subset of the state dict?
        # FIXME: Should these be options in the method?

        # Convert nodes from string to int
        nodes_list = [int(ni) for ni in nodes_listo]
        img_pos = 'bc'  # position of the glyph in the node
        subcluster_font = 'DejaVu Sans Bold'
        node_shape = 'ellipse'
        clr_map = 'rainbow_r'
        nde_font_color = 'Black'
        hex_transparency = '80'

        # Try to make a nested graph:
        G = pgv.AGraph(strict=mono_edge,
                       fontname=subcluster_font,
                       splines=True,
                       directed=True,
                       concentrate=concentrate,
                       constraint=constraint,
                       rank=rank,
                       dpi=300)

        cmap = colormaps[clr_map]

        if node_colors is None:
            norm = colors.Normalize(vmin=0, vmax=self._solsM_all.shape[1] +1)
        else:
            norm = colors.Normalize(vmin=np.min(node_colors),
                                    vmax=np.max(node_colors))

        # Add all the nodes:
        for nde_i in nodes_list:
            nde_lab = nde_i
            nde_index = nodes_list.index(nde_i)

            if node_colors is None:
                nde_color = colors.rgb2hex(cmap(norm(nde_lab)))
            else:
                nde_color = colors.rgb2hex(cmap(norm(node_colors[nde_lab])))

            nde_color += hex_transparency  # add some transparancy to the node

            char_i = charM_all[nde_i] # Get the stability characterization for this state

            G.add_node(nde_i,
                           label=f'State {nde_lab}',
                           labelloc='t',
                           image=self._node_image_dict[char_i],
                           imagepos=img_pos,
                           shape=node_shape,
                           fontcolor=nde_font_color,
                           style='filled',
                           fillcolor=nde_color)


        # Add all the edges:
        for nde_i, nde_j, trans_ij in edges_list:
            G.add_edge(nde_i, nde_j, label=trans_ij, fontsize=fontsize)

        if save_file is not None:
            G.layout(prog=graph_layout)
            G.draw(save_file)

        return G

    def plot_state_perturbation_network(self,
                                       pert_edges_set: set,
                                       charM_all: list | ndarray,
                                       nodes_listo: list|ndarray,
                                       save_file: str|None = None,
                                       graph_layout: str = 'dot',
                                       mono_edge: bool=False,
                                       rank: str = 'same',
                                       constraint: bool=False,
                                       concentrate: bool=True,
                                       fontsize: float = 18.0,
                                       node_colors: list | None = None
                                        ):
        '''
        This network plotting and generation function is based on the concept
        that an input node state can be associated with several gene network
        states if the network has multistability. Here we create a graph with
        subgraphs, where each subgraph represents the possible states for a
        held input node state. In the case of multistability, temporary
        perturbations to the held state can result in transitions between
        the multistable state (resulting in a memory and path-dependency). The
        graph indicates which input signal perturbation leads to which state
        transition via the edge label. Input signal states are represented as
        integers, where the integer codes for a binary bit string of signal state values.

        Parameters
        ----------
        pert_edges_set : set
            Tuples of state i, state j, perturbation input integer, base input integer, generated
            by create_transition_network.

        states_dict: dict
            Dictionary of states and their stability characterization tags for each input signal set.

        nodes_list : list|None = None
            A list of nodes to include in the network. This is useful to filter out inaccessible states,
            if desired.

        save_file : str|None = None
            A file to save the network image to. If None, no image is saved.

        graph_layout : str = 'dot'
            Layout for the graph when saving to image.

        '''


        nodes_list = [int(ni) for ni in nodes_listo] # convert nodes from string to int

        img_pos = 'bc'  # position of the glyph in the node
        subcluster_font = 'DejaVu Sans Bold'
        node_shape = 'ellipse'
        clr_map = 'rainbow_r'
        nde_font_color = 'Black'
        hex_transparency = '80'

        # Make a nested graph with compound=True keyword:
        G = pgv.AGraph(strict=mono_edge,
                       fontname=subcluster_font,
                       splines=True,
                       directed=True,
                       concentrate=concentrate,
                       constraint=constraint,
                       compound=True,
                       rank=rank,
                       dpi=300)

        cmap = colormaps[clr_map]

        if node_colors is None:
            norm = colors.Normalize(vmin=0, vmax=self._solsM_all.shape[1] +1)
        else:
            norm = colors.Normalize(vmin=np.min(node_colors), vmax=np.max(node_colors))

        for st_i, st_f, i_pert, i_base in pert_edges_set:
            # Add in a subgraph box for the "held" input node state:
            Gsub = G.add_subgraph(name=f'cluster_{i_base}', label=f'Held at I{i_base}')

            # next add-in nodes for the initial state:
            nde_i_name = f'{st_i}.{i_base}' # node name is in terms of the subgraph box index
            nde_i_lab = f'State {st_i}'

            if node_colors is None:
                nde_i_color = colors.rgb2hex(cmap(norm(st_i)))
            else:
                nde_i_color = colors.rgb2hex(cmap(norm(node_colors[st_i])))

            nde_i_color += hex_transparency  # add some transparency to the node

            chr_i = charM_all[st_i]

            Gsub.add_node(nde_i_name,
                          label=nde_i_lab,
                          labelloc='t',
                          image=self._node_image_dict[chr_i],
                          imagepos=img_pos,
                          shape=node_shape,
                          fontcolor=nde_font_color,
                          style='filled',
                          fillcolor=nde_i_color
                          )

            # ...and for the final state:
            nde_f_name = f'{st_f}.{i_base}' # node name is in terms of the subgraph box index
            nde_f_lab = f'State {st_f}'
            nde_f_color = colors.rgb2hex(cmap(norm(st_f)))
            nde_f_color += hex_transparency  # add some transparency to the node
            chr_f = charM_all[st_f]

            Gsub.add_node(nde_f_name,
                          label=nde_f_lab,
                          labelloc='t',
                          image=self._node_image_dict[chr_f],
                          imagepos=img_pos,
                          shape=node_shape,
                          fontcolor=nde_font_color,
                          style='filled',
                          fillcolor=nde_f_color
                          )

            Gsub.add_edge(nde_i_name, nde_f_name, label=f'I{i_pert}', fontsize=fontsize)

        if save_file is not None:
            G.layout(prog=graph_layout)
            G.draw(save_file)

        return G


    def plot_time_trajectory(self,
                             c_time: ndarray,
                             tvectr: ndarray|list,
                             phase_inds: ndarray|list,
                             matched_states: ndarray|list,
                             charM_all: ndarray|list,
                             gene_plot_inds: list|None=None,
                             figsize: tuple = (10, 4),
                             state_label_offset: float = 0.02,
                             glyph_zoom: float=0.15,
                             glyph_alignment: tuple[float, float]=(-0.0, -0.15),
                             fontsize: str='medium',
                             save_file: str|None = None,
                             legend: bool=True,
                             ):
        '''

        '''

        if gene_plot_inds is None:
            main_c = c_time[:, self._pnet.noninput_node_inds]
        else:
            main_c = c_time[:, gene_plot_inds]

        N_plot_genes = main_c.shape[1]

        # Resize the figure to fit the panel of plotted genes:
        fig_width = figsize[0]
        fig_height = figsize[1]
        figsize = (fig_width, fig_height*N_plot_genes)

        cmap = plt.get_cmap("tab10")

        fig, axes = plt.subplots(N_plot_genes, 1, figsize=figsize, sharex=True, sharey=True)
        for ii, cc in enumerate(main_c.T):
            gene_lab = f'Gene {ii}'
            lineplt = axes[ii].plot(tvectr, cc, linewidth=2.0, label=gene_lab, color=cmap(ii))  # plot the time series
            # annotate the plot with the matched state:
            for (pi, pj), stateio in zip(phase_inds, matched_states):
                statei = stateio

                char_i = charM_all[stateio] # We want the state characterization to go to the full state system
                char_i_fname = self._node_image_dict[char_i]
                logo = image.imread(char_i_fname)
                imagebox = OffsetImage(logo, zoom=glyph_zoom)
                pmid = pi
                tmid = tvectr[pmid]
                cc_max = np.max(cc[pi:pj])
                cmid = cc_max + state_label_offset

                axes[ii].text(tmid, cmid, f'State {statei}', fontsize=fontsize)

                ab = AnnotationBbox(imagebox,
                                    (tmid, cmid),
                                    frameon=False,
                                    box_alignment=glyph_alignment)
                axes[ii].add_artist(ab)

                axes[ii].spines['top'].set_visible(False)
                axes[ii].spines['right'].set_visible(False)

                axes[ii].set_ylabel('Expression Probability')

                if legend:
                    axes[ii].legend(frameon=False)

        axes[-1].set_xlabel('Time')

        if save_file is not None:
            plt.savefig(save_file, dpi=300, transparent=True, format='png')

        return fig, axes

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

    def _get_input_signals_from_label_dict(self, sig_test_set: ndarray | list):
        '''

        '''
        # Would be very useful to have a lookup dictionary between the integer input
        # state label and the original signals tuple:
        input_int_to_signals = {}

        for int_label, input_sigs in enumerate(sig_test_set):
            # int_label = self._get_integer_label(input_sigs)
            input_int_to_signals[f'I{int_label}'] = tuple(input_sigs)

        return input_int_to_signals

    def _order_states_by_distance(self, solsM_all, charM_all):
        '''
        Re-arrange the supplied solution matrix so that the states are
        progressively closer to one another, in order to see a more
        logical transition through the network with perturbation.
        '''
        zer_sol = np.zeros(solsM_all[:, 0].shape)
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

    # def _get_integer_label(self, sig_set: tuple|list|ndarray) -> int:
    #     '''
    #     Given a list of digits representing a bit string
    #     (i.e. a list of values close to zero or 1), this method
    #     treats the list as a binary bit-string and returns the
    #     base-2 integer representation of the bit-string.
    #
    #     Parameters
    #     ----------
    #     sig_set : list[float|int]
    #         The list containing floats or ints that are taken to represent
    #         a bit string.
    #
    #     Returns
    #     -------
    #     An integer representation of the binary bit-string.
    #
    #     '''
    #     base_str = ''
    #     for sigi in sig_set:
    #         base_str += str(int(sigi))
    #     return int(base_str, 2)

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
            errM.append(np.sqrt(np.sum(sdiff ** 2)))
        errM = np.asarray(errM)
        state_best_match = (errM == errM.min()).nonzero()[0][0]

        return state_best_match, errM[state_best_match]

    def plot_input_words_array(self,
                        sig_test_set: ndarray,
                        gene_list: list|ndarray,
                        figsave: str | None = None,
                        cmap: str | None =None,
                        save_format: str='png',
                        figsize: tuple=(10,10)):
        '''

        '''

        if cmap is None:
            cmap = 'magma'

        state_labels = [f'I{i}' for i in range(sig_test_set.shape[0])]

        gene_labels = np.asarray(gene_list)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(sig_test_set, cmap=cmap)

        ax.set_xticks(np.arange(len(gene_labels)), labels=gene_labels)
        ax.set_yticks(np.arange(len(state_labels)), labels=state_labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        fig.colorbar(im, label='Expression Level')

        if figsave is not None:
            plt.savefig(figsave, dpi=300, transparent=True, format=save_format)

        return fig, ax

    # def _get_unique_sol_dict(self, solsM_all: ndarray, match_tol: float=0.05, N_round_sol: int=1):
    #     '''
    #     Returns a dictionary that maps an index in solsM_all to an index to the unique
    #     sols in solsM_all with respect to the noninput node indes.
    #
    #     Parameters
    #     ----------
    #     solsM_all : ndarray
    #         An array of concentrations in rows and unique equilibrium states in columns.
    #
    #     match_tol : float=0.05
    #         The tolerance, above which a state is taken to not be a match.
    #
    #     '''
    #     # The sols that are unique with respect to the non-input nodes:
    #     solsM_all_unique, unique_inds = np.unique(np.round(solsM_all[self._pnet.noninput_node_inds],
    #                                                        N_round_sol), axis=1, return_index=True)
    #
    #     N_unique_sols = len(unique_inds)
    #
    #     unique_sol_dict = {}
    #     inv_unique_sols_dict = {}
    #     for si, soli in enumerate(solsM_all.T):
    #         # Find a match between soli and a solution in solsM_all_unique:
    #         state_match, match_error = self._find_state_match(solsM_all_unique,
    #                                                            soli[self._pnet.noninput_node_inds])
    #         if match_error < match_tol:
    #             unique_sol_dict[si] = state_match
    #             inv_unique_sols_dict[state_match] = si # record the inverse mapping as well
    #
    #         else:
    #             raise Exception("State not found!")
    #
    #     return unique_sol_dict, inv_unique_sols_dict, N_unique_sols
