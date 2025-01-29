#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds and plots a state transition diagram from a solution
set and corresponding network model based on a Boolean network formalism.
'''

import os
import itertools
import copy
import numpy as np
import networkx as nx
import pygraphviz as pgv
from cellnition.science.network_models.boolean_networks import BooleanNet
from cellnition.science.network_models.network_enums import EquilibriumType
from cellnition._util.path.utilpathmake import FileRelative
from cellnition._util.path.utilpathself import get_data_png_glyph_stability_dir
from collections import OrderedDict
from numpy import ndarray
from cellnition.types import NumpyTrue
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from networkx import MultiDiGraph
from pygraphviz import AGraph

class BoolStateMachine(object):
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

    def __init__(self, bnet: BooleanNet):
        '''
        Initialize the BoolStateMachine.

        Parameters
        ----------
        bnet : BooleanNet
            An instance of Boolean network.

        solsM : ndarray
            A set of unique steady-state solutions from the GeneNetworkModel.
            These will be the states of the StateMachine.
        '''

        self._bnet = bnet
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

    def run_state_machine(self,
                          verbose: bool=True,
                          save_graph_file: str|None =None,
                          save_transition_net_image: str | None = None,
                          save_perturbation_net_image: str|None = None,
                          graph_layout: str = 'dot',
                          remove_inaccessible_states: bool = True,
                          search_main_nodes_only: bool = False
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
         sig_test_set) = self.steady_state_solutions_search(verbose=verbose,
                                                            search_main_nodes_only=search_main_nodes_only
                                                            )

        # save entities to the object:
        self.solsM_all = solsM_all
        self.charM_all = charM_all
        self.sols_list = sols_list
        self.states_dict = states_dict
        self.sig_test_set = sig_test_set # should be called "input space states"

        # Create the edges of the transition network:
        transition_edges_set, pert_edges_set, G_nx = self.create_transition_network(states_dict,
                                                                                    sig_test_set,
                                                                                    solsM_all,
                                                                     verbose=verbose,
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
                                                           save_file=save_perturbation_net_image,
                                                          graph_layout=graph_layout,
                                                        )

        if save_transition_net_image is not None: # save an image of the network to file:
            # get nodes and edges list:
            nodes_list = sorted(G_nx.nodes())
            edges_list = list(G_nx.edges)

            G_gv = self.plot_state_transition_network(nodes_list,
                                                      edges_list,
                                                      solsM_all,
                                                      charM_all,
                                                      save_file=save_transition_net_image,
                                                      graph_layout=graph_layout
                                                      )

        return G_nx


    def steady_state_solutions_search(self,
                                      verbose: bool=True,
                                      search_main_nodes_only: bool = False,
                                      n_max_steps: int = 10
                                      ):
        '''
        Search through all possible combinations of signal node values
        and collect and identify all equilibrium points of the system.

        '''
        if self._bnet._A_bool_f is None:
            raise Exception("The BooleanNetwork object needs to have an analytical model constructed using"
                            "the build_boolean_model method.")

        sig_lin = [0, 1]

        sig_lin_set = [sig_lin for i in self._bnet.input_node_inds]

        sigGrid = np.meshgrid(*sig_lin_set)

        N_vocab = len(sigGrid[0].ravel())

        sig_test_set = np.zeros((N_vocab, len(self._bnet.input_node_inds)))

        for i, sigM in enumerate(sigGrid):
            sig_test_set[:, i] = sigM.ravel()

        # solsM_allo = np.zeros((self._bnet.N_nodes, 1))
        # charM_allo = [EquilibriumType.undetermined]
        solsM_allo = None
        charM_allo = []
        sols_list = []

        for sigis in sig_test_set:
            if verbose:
                print(f'Signals: {sigis}')
            sols_M, sols_char = self._bnet.solve_system_equms(
                                                        self._bnet._A_bool_f,
                                                        constraint_inds=None,
                                                        constraint_vals=None,
                                                        signal_constr_vals=sigis.tolist(),
                                                        search_main_nodes_only=search_main_nodes_only,
                                                        n_max_steps=n_max_steps,
                                                        verbose=False
                                                        )
            if solsM_allo is None:
                solsM_allo = sols_M
            else:
                solsM_allo = np.hstack((solsM_allo, sols_M))  # append all unique sols
            charM_allo.extend(sols_char.tolist())  # append the sol stability characterization tags
            sols_list.append(sols_M)
            if verbose:
                print(sols_M)
                print('----')

        # Eliminate duplicate states, but stack on strings of the eqm char
        # so we don't lose states with the same values but with different eq'm:
        chrm = [eqt.name for eqt in charM_allo]
        checkM = np.vstack((solsM_allo[self._bnet.noninput_node_inds, :], chrm))
        _, inds_solsM_all_unique = np.unique(checkM, return_index=True, axis=1)
        solsM_all = solsM_allo[:, inds_solsM_all_unique]
        charM_all = np.asarray(charM_allo)[inds_solsM_all_unique]

        # Order states by distance from the zero vector
        solsM_all, charM_all = self._order_states_by_distance(solsM_all, charM_all)

        states_dict = OrderedDict()
        for sigi in sig_test_set:
            states_dict[tuple(sigi)] = {'States': [], 'Stability': []}

        for sigi, state_subseto in zip(sig_test_set, sols_list):
            state_subset = state_subseto[self._bnet.noninput_node_inds, :]
            for target_state in state_subset.T.tolist():
                state_match_index, err_match = self._find_state_match(solsM_all[self._bnet.noninput_node_inds, :],
                                                                      target_state)
                if state_match_index not in states_dict[tuple(sigi)]['States']:
                    states_dict[tuple(sigi)]['States'].append(state_match_index)
                    states_dict[tuple(sigi)]['Stability'].append(charM_all[state_match_index])

        return solsM_all, charM_all, sols_list, states_dict, sig_test_set

    def create_transition_network(self,
                                  states_dict: dict,
                                  sig_test_set: list|ndarray,
                                  solsM_allo: ndarray,
                                  verbose: bool = True,
                                  remove_inaccessible_states: bool=False,
                                  save_graph_file: str|None = None,
                                  n_max_steps: int=10
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

        sig_inds = self._bnet.input_node_inds

        transition_edges_set = set()
        perturbation_edges_set = set()

        num_step = 0

        # We want to step through all 'held' signals and potentially multistable states:
        # The "base" refers to the base context of the system:
        for base_input_label, (sig_base_set, sc_dict) in enumerate(states_dict.items()):

            states_set = sc_dict['States']

            # Get an integer label for the 'bitstring' of signal node inds defining the base:
            # base_input_label = self._get_integer_label(sig_base_set)
            # We want to use each state in states_set as the initial condition:
            for si in states_set:
            # for si, cvect_co in enumerate(solsM_all.T): # step through all stable states

                if verbose:
                    print(f"Testing State {si} in held context I{base_input_label}...")

                cvect_co = solsM_all[:, si]

                # Initial state vector, which will be held at the base_input context
                cvect_co[sig_inds] = sig_base_set
                # ensure that this initial state is indeed stable in the context:
                cvect_c, char_c = self._bnet.net_state_compute(cvect_co,
                                                               self._bnet._A_bool_f,
                                                               n_max_steps=n_max_steps,
                                                               verbose=False,
                                                               constraint_inds=self._bnet.input_node_inds,
                                                               constraint_vals=list(sig_base_set),
                                                               )

                initial_state, match_error_initial = self._find_state_match(solsM_all[self._bnet.noninput_node_inds, :],
                                                                      cvect_c[self._bnet.noninput_node_inds])

                if match_error_initial > 0.0:  # if held state is unmatched, flag it with a nan
                    solsM_all = np.column_stack((solsM_all, cvect_c))
                    initial_state = solsM_all.shape[1] - 1

                    if verbose:
                        print(f'WARNING: Initial state not found; adding new state {initial_state} to the solution set...')

                # Add this transition to the state transition diagram:
                transition_edges_set.add((si, initial_state, base_input_label))

                # We then step through all possible perturbation signals that act on the state in the set:
                for pert_input_label, sig_pert_set in enumerate(sig_test_set):
                    if verbose:
                        print(f"--- Step: {num_step} ---")
                        print(f'...State {si} to {initial_state} via I{base_input_label}...')

                    cvect_ho = cvect_c.copy()
                    cvect_ho[sig_inds] = sig_pert_set # apply the perturbation to the state
                    cvect_h, char_h = self._bnet.net_state_compute(cvect_ho,
                                                               self._bnet._A_bool_f,
                                                               n_max_steps=n_max_steps,
                                                               verbose=False,
                                                               constraint_inds=self._bnet.input_node_inds,
                                                               constraint_vals=list(sig_pert_set),
                                                               )

                    # FIXME: the find_state_match method should use the equm' char as well as the state values!
                    # match the network state to one that only involves the hub nodes:
                    held_state, match_error_held = self._find_state_match(solsM_all[self._bnet.noninput_node_inds, :],
                                                                                cvect_h[self._bnet.noninput_node_inds])

                    if match_error_held > 0.0: # if held state is unmatched, flag it with a nan
                        solsM_all = np.column_stack((solsM_all, cvect_h))
                        held_state = solsM_all.shape[1] - 1

                        if verbose:
                            print(f'WARNING: Held state not found; adding new state {held_state} to the solution set...')

                    transition_edges_set.add((initial_state, held_state, pert_input_label))
                    if verbose:
                        print(f'...State {initial_state} to {held_state} via I{pert_input_label}...')

                    # Next, re-apply the initial context input state to the held state and see what final state results:
                    cvect_fo = cvect_h.copy()
                    cvect_fo[sig_inds] = sig_base_set

                    # find the stable state that results from re-applying the context input state to the held state:
                    cvect_f, char_f = self._bnet.net_state_compute(cvect_fo,
                                                               self._bnet._A_bool_f,
                                                               n_max_steps=n_max_steps,
                                                               verbose=False,
                                                               constraint_inds=self._bnet.input_node_inds,
                                                               constraint_vals=list(sig_base_set),
                                                               )

                    final_state, match_error_final = self._find_state_match(solsM_all[self._bnet.noninput_node_inds, :],
                                                                          cvect_f[self._bnet.noninput_node_inds])
                    # held_state, match_error_held = self._find_state_match(solsM_all, c_held)

                    if match_error_held > 0.0: # if state is unmatched, flag it
                        solsM_all = np.column_stack((solsM_all, cvect_f))
                        final_state = solsM_all.shape[1] -1

                        if verbose:
                            print(f'WARNING: Final state not found; adding new state {final_state} to the solution set...')

                    transition_edges_set.add((held_state, final_state, base_input_label))
                    if verbose:
                        print(f'...State {held_state} to {final_state} via I{base_input_label}')

                    # Look for change in the system from initial to final state:

                    if initial_state != final_state:  # add this to the perturbed transitions:
                        perturbation_edges_set.add((initial_state, final_state, pert_input_label, base_input_label))

                        if verbose:
                            print(f'Event-driven transition identified from State {initial_state} to {final_state} via '
                                  f'event I{pert_input_label} under context I{base_input_label}')

                    num_step += 1

                    # if verbose:
                    #     # print(f'Match errors {match_error_initial, match_error_held, match_error_final}')
                    #     print('------')

        # The first thing we do after the construction of the
        # transition edges set is make a multidigraph and
        # use networkx to pre-process & simplify it, removing inaccessible states
        # (states with no non-self input degree)

        self._solsM_all = solsM_all
        self._states_dict = states_dict_2
        self._sig_test_set = sig_test_set

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

        self._charM_all = charM_all
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

            char_i = charM_all[nde_i].name # Get the stability characterization for this state

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

            chr_i = charM_all[st_i].name

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
            chr_f = charM_all[st_f].name

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
                    np.sum((solsM_all[self._bnet.noninput_node_inds, i] -
                            solsM_all[self._bnet.noninput_node_inds, j]) ** 2))
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
            dist_list.append(np.sqrt(np.sum((zer_sol[self._bnet.noninput_node_inds] -
                                             soli[self._bnet.noninput_node_inds]) ** 2)))

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


