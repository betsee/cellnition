#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds and plots a state transition diagram from a solution
set and corresponding GeneNetworkModel.
'''
import os
import itertools
from collections import OrderedDict
import numpy as np
from numpy import ndarray
from matplotlib import colors
from matplotlib import colormaps
import networkx as nx
from networkx import MultiDiGraph
from cellnition.science.network_models.probability_networks import ProbabilityNet
from cellnition.science.network_models.network_enums import EquilibriumType
import pygraphviz as pgv
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
        # FIXME: this needs to be internal to the project
        glyph_path = '/home/pietakio/Dropbox/Levin_2023/CellnitionAssets/StabilityGlyphs/large'
        attractor_fname = os.path.join(glyph_path, 'glyph_attractor.png')
        limitcycle_fname = os.path.join(glyph_path, 'glyph_limit_cycle.png')
        saddle_fname = os.path.join(glyph_path, 'glyph_saddle.png')

        # Associate each equilibrium type with an image file
        self._node_image_dict = {EquilibriumType.attractor.name: attractor_fname,
                           EquilibriumType.limit_cycle.name: limitcycle_fname,
                           EquilibriumType.saddle.name: saddle_fname
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
                                      ):
        '''
        Search through all possible combinations of signal node values
        and collect and identify all equilibrium points of the system.

        '''
        # FIXME: do we want this to have additional node constraints?

        sig_lin = [1.0e-6, 1.0]
        sig_lin_set = [sig_lin for i in self._pnet.signal_node_inds]

        sigGrid = np.meshgrid(*sig_lin_set)

        N_vocab = len(sigGrid[0].ravel())

        sig_test_set = np.zeros((N_vocab, len(self._pnet.signal_node_inds)))

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
        _, inds_solsM_all_unique = np.unique(np.round(solsM_all, 1)[self._pnet.nonsignal_node_inds, :], axis=1,
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

        # set of all states referencing only the hub nodes; rounded to one decimal:
        state_set = np.round(solsM_all[self._pnet.nonsignal_node_inds, :].T, 1).tolist()

        states_dict = OrderedDict()
        for sigi in sig_test_set:
            states_dict[tuple(sigi)] = {'States': [], 'Stability': []}

        for sigi, state_subseto in zip(sig_test_set, sols_list):
            state_subset = state_subseto[self._pnet.nonsignal_node_inds, :]
            for target_state in np.round(state_subset, 1).T.tolist():
                if target_state in state_set:
                    state_match_index = state_set.index(target_state)
                    states_dict[tuple(sigi)]['States'].append(state_match_index)
                    states_dict[tuple(sigi)]['Stability'].append(charM_all[state_match_index])
                else:
                    print('match not found!')
                    states_dict[tuple(sigi)]['States'].append(np.nan)
                    states_dict[tuple(sigi)]['Stability'].append(np.nan)

        return solsM_all, charM_all, sols_list, states_dict

    def infer_state_transition_network(self,
                                       states_dict: dict,
                                       save_graph_filename: str|None=None,
                                       graph_layout: str='dot')->AGraph:
        '''
        Using a states_dict prepared in a state space search, infer the
        state transition network based on overlapping nonsignal node indices
        between two sets.

        '''
        # Try to make a nested graph:
        G = pgv.AGraph(strict=False,
                       splines=True,
                       directed=True,
                       concentrate=False,
                       compound=True,
                       dpi=300)

        # We first need to make all the subgraphs:
        for trans_sigs_i, states_dict_i in states_dict.items():

            states_set_i = states_dict_i['States']
            states_char_i = states_dict_i['Stability']

            trans_label_io = ''
            for ii in trans_sigs_i:
                trans_label_io += str(int(ii))

            trans_label_i = int(trans_label_io, 2)

            G.add_subgraph(name=f'cluster_{trans_label_i}', label=f'Held at S{trans_label_i}')

        # Then get the way this specific graph will order them:
        subg_list = [subg.name for subg in G.subgraphs()]

        for i, (trans_sigs_i, states_dict_i) in enumerate(states_dict.items()):

            states_set_i = states_dict_i['States']
            states_char_i = states_dict_i['Stability']

            trans_label_io = ''
            for ii in trans_sigs_i:
                trans_label_io += str(int(ii))

            trans_label_i = int(trans_label_io, 2)

            G_sub = G.subgraphs()[subg_list.index(f'cluster_{trans_label_i}')]

            if len(states_set_i) > 1:

                for trans_sigs_j, states_dict_j in states_dict.items():

                    states_set_j = states_dict_j['States']
                    states_char_j = states_dict_j['Stability']

                    trans_label_jo = ''
                    for ii in trans_sigs_j:
                        trans_label_jo += str(int(ii))
                    trans_label_j = int(trans_label_jo, 2)

                    if trans_sigs_i != trans_sigs_j:
                        shared_states = np.intersect1d(states_set_i, states_set_j)

                        if len(shared_states) == 1:

                            sj = shared_states[0]
                            nde_j = f'{trans_label_i}.{sj}'
                            G_sub.add_node(nde_j, label=f'State {sj}')

                            for si in states_set_i:
                                nde_i = f'{trans_label_i}.{si}'
                                G_sub.add_node(nde_i, label=f'State {si}')
                                G_sub.add_edge(nde_i, nde_j, label=f'S{trans_label_j}')

                        elif len(
                                shared_states) > 1:  # allow for a 3rd level of hierarchy...we don't know what it means yet...
                            for sj in shared_states:
                                nde_j = f'{trans_label_i}.{sj}'
                                G_sub.add_node(nde_j, label=f'State {sj}')

                                for si in states_set_i:
                                    nde_i = f'{trans_label_i}.{si}'
                                    G_sub.add_node(nde_i, label=f'State {si}')
                                    G_sub.add_edge(nde_i, nde_j, label=f'S{trans_label_j}')

            else:
                for si in states_set_i:
                    nde_i = f'{trans_label_i}.{si}'
                    G_sub.add_node(nde_i, label=f'State {si}')

        # Finally, we add in transitions between the "held" states:
        for nde_i in G.nodes():
            assert len(nde_i) >= 3
            ni = nde_i[2:]
            for nde_j in G.nodes():
                nj = nde_j[2:]
                if ni == nj and nde_i != nde_j:
                    G.add_edge(nde_i, nde_j)

        if save_graph_filename is not None:
            G.layout(prog=graph_layout)
            G.draw(save_graph_filename)

        return G


    def plot_inferred_state_transition_network(self,
                                              states_dict: dict,
                                              solsM_all: ndarray|list,
                                              save_file: str|None = None,
                                              graph_layout: str='dot'
                                              ):
        '''

        '''
        # FIXME: we probably also want the option to just plot a subset of the state dict?
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
                       compound=True,
                       dpi=300)

        cmap = colormaps[clr_map]
        norm = colors.Normalize(vmin=0, vmax=solsM_all.shape[1])

        # We first need to make all the subgraphs:
        for trans_sigs_i, states_dict_i in states_dict.items():

            trans_label_io = ''
            for ii in trans_sigs_i:
                trans_label_io += str(int(ii))

            trans_label_i = int(trans_label_io, 2)

            G.add_subgraph(name=f'cluster_{trans_label_i}', label=f'Held at S{trans_label_i}')

        # Then get the way this specific graph will order them:
        subg_list = [subg.name for subg in G.subgraphs()]

        for trans_sigs_i, states_dict_i in states_dict.items():

            states_set_i = states_dict_i['States']
            states_char_i = states_dict_i['Stability']

            trans_label_io = ''
            for ii in trans_sigs_i:
                trans_label_io += str(int(ii))

            trans_label_i = int(trans_label_io, 2)

            G_sub = G.subgraphs()[subg_list.index(f'cluster_{trans_label_i}')]

            if len(states_set_i) > 1:

                for trans_sigs_j, states_dict_j in states_dict.items():

                    states_set_j = states_dict_j['States']
                    states_char_j = states_dict_j['Stability']

                    trans_label_jo = ''
                    for ii in trans_sigs_j:
                        trans_label_jo += str(int(ii))
                    trans_label_j = int(trans_label_jo, 2)

                    if trans_sigs_i != trans_sigs_j:
                        shared_states = np.intersect1d(states_set_i, states_set_j)

                        if len(shared_states) >= 1:  # allow for a 3rd level of hierarchy...we don't know what it means yet...
                            for sj in shared_states:
                                lini_sj = states_set_j.index(sj)  # get the index of sj in the original list
                                char_sj = states_char_j[lini_sj]  # so we can get the equ'm char of sj

                                nde_j = f'{trans_label_i}.{sj}'

                                nde_color = colors.rgb2hex(cmap(norm(sj)))
                                nde_color += '80'  # add some transparancy to the node

                                G_sub.add_node(nde_j,
                                               label=f'State {sj}',
                                               labelloc='t',
                                               image=self._node_image_dict[char_sj],
                                               imagepos=img_pos,
                                               shape=node_shape,
                                               fontcolor=nde_font_color,
                                               style='filled',
                                               fillcolor=nde_color)

                                for si in states_set_i:
                                    lini_si = states_set_i.index(si)  # get the index of si in the original list
                                    char_si = states_char_i[lini_si]  # so we can get the equ'm char of si

                                    nde_color = colors.rgb2hex(cmap(norm(si)))
                                    nde_color += hex_transparency  # add some transparancy to the node

                                    nde_i = f'{trans_label_i}.{si}'
                                    G_sub.add_node(nde_i,
                                                   label=f'State {si}',
                                                   labelloc='t',
                                                   image=self._node_image_dict[char_si],
                                                   imagepos=img_pos,
                                                   shape=node_shape,
                                                   fontcolor=nde_font_color,
                                                   style='filled',
                                                   fillcolor=nde_color)

                                    G_sub.add_edge(nde_i, nde_j, label=f'S{trans_label_j}')

            else:
                for si in states_set_i:
                    lini_si = states_set_i.index(si)  # get the index of si in the original list
                    char_si = states_char_i[lini_si]  # so we can get the equ'm char of si

                    nde_color = colors.rgb2hex(cmap(norm(si)))
                    nde_color += '80'  # add some transparancy to the node

                    nde_i = f'{trans_label_i}.{si}'
                    G_sub.add_node(nde_i,
                                   label=f'State {si}',
                                   labelloc='t',
                                   image=self._node_image_dict[char_si],
                                   imagepos=img_pos,
                                   shape=node_shape,
                                   fontcolor=nde_font_color,
                                   style='filled',
                                   fillcolor=nde_color)

        # Finally, we add in transitions between the "held" states:
        for nde_i in G.nodes():
            assert len(nde_i) >= 3
            ni = nde_i[2:]
            for nde_j in G.nodes():
                nj = nde_j[2:]
                if ni == nj and nde_i != nde_j:
                    G.add_edge(nde_i, nde_j)

        if save_file is not None:
            G.layout(prog=graph_layout)
            G.draw(save_file)

        return G


    def _order_states_by_distance(self):
        '''
        Re-arrange the supplied solution matrix so that the states are
        progressively closer to one another, in order to see a more
        logical transition through the network with perturbation.
        '''
        pass

    def find_state_match(self,
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


    def create_transition_network(self,
                                  signal_node_inds: list[int],
                                  dt: float = 1.0e-3,
                                  tend: float = 100.0,
                                  sig_tstart: float = 33.0,
                                  sig_tend: float = 66.0,
                                  sig_base: float = 0.0,
                                  sig_active: float = 1.0,
                                  delta_window: float = 1.0,
                                  dt_samp: float=0.15,
                                  verbose: bool = True,
                                  tol: float = 1.0e-6,
                                  do_combos: bool=False,
                                  constrained_inds: list | None = None,
                                  constrained_vals: list | None = None,
                                  d_base: float = 1.0,
                                  n_base: float = 15.0,
                                  beta_base: float = 0.25
                                  ):
        '''
        Build a state transition matrix/diagram by starting the system
        in different states and seeing which state it ends up in after
        a time simulation. This method iterates through all 'signal'
        nodes of a network and sets them to the sigmax level, harvesting
        the new steady-state reached after perturbing the network.

        Parameters
        ----------
        signal_node_inds : list[int]
            List specifing the nodes to be peturbed in the creation of this state
            transition network.

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

        do_combos : bool = False
            Consider combinations of all signal nodes in the network (e.g. s2, s0 + s1, s0 + s1 +s2)
            as possible transitions for states.

        '''
        c_zeros = np.zeros(self._pnet.N_nodes)  # start everything out low

        # Create a steady-state solutions matrix that is stacked with the
        # 'zero' or 'base' state:
        solsM_with0 = np.column_stack((c_zeros, self._solsM))

        G_states = MultiDiGraph()
        # G_states = DiGraph()

        if do_combos is False: # Just do perturbations in terms of each signal node
            signal_inds = [[sigi] for sigi in signal_node_inds]
        else: # Do perturbations as all possible combinations (without replacement) or each signal node
            lensig = len(signal_node_inds)
            signal_inds = []
            for i in range(0, lensig):
                signal_inds.extend([list(sigis) for sigis in itertools.combinations(signal_node_inds, i + 1)])

        # Make string labels for each of the signals:
        signal_labels = []
        for sigi in signal_inds:
            if len(sigi) > 1:
                str_lab = ''
                for i, nss in enumerate(sigi):
                    ss = self._pnet.nodes_list[nss]
                    if i == 0:
                        str_lab += f'{ss}'
                    else:
                        str_lab += f'*{ss}'
                signal_labels.append(str_lab)
            else:
                ss = self._pnet.nodes_list[sigi[0]]
                signal_labels.append(f'{ss}')

        for stateio, cvecto in enumerate(solsM_with0.T):  # start the system off in each of the states

            # For each signal in the network:
            for si, sigi in enumerate(signal_inds):
                cvecti = cvecto.copy()  # reset the state to the desired starting state
                sig_times = [(sig_tstart, sig_tend) for ss in sigi]
                sig_mags = [(sig_base, sig_active) for ss in sigi]

                # Run the time simulation:
                concs_time, tvect = self._pnet.run_time_sim(tend,
                                                            dt,
                                                            cvecti,
                                                            sigi,
                                                            sig_times,
                                                            sig_mags,
                                                            dt_samp=dt_samp,
                                                            constrained_inds=constrained_inds,
                                                            constrained_vals=constrained_vals,
                                                            d_base=d_base,
                                                            n_base=n_base,
                                                            beta_base=beta_base
                                                            )

                it_30low = (tvect <= sig_tstart - delta_window).nonzero()[0]
                it_30high = (tvect >= sig_tstart - 2 * delta_window).nonzero()[0]
                it_30 = np.intersect1d(it_30low, it_30high)[0]

                concs_before = concs_time[it_30, :]
                concs_after = concs_time[-1, :]

                if stateio == 0 and si == 0:  # If we're on the zeros vector we've transitioned from {0} to some new state:
                    statejo, errio = self.find_state_match(solsM_with0, concs_before)
                    if errio < tol:
                        G_states.add_edge(0, statejo, transition=-1)

                    else:  # otherwise it's not a match so add the new state to the system:
                        solsM_with0 = np.column_stack((solsM_with0, concs_before))
                        statejo, errio = self.find_state_match(solsM_with0, concs_before)
                        G_states.add_edge(0, statejo, transition=-1)

                    if verbose:
                        print(f'From state 0 spontaneously to state {statejo}')

                statei, erri = self.find_state_match(solsM_with0, concs_before)
                statej, errj = self.find_state_match(solsM_with0, concs_after)

                # FIXME: this doesn't work to add in states as we go -- results in
                # a mess!
                if erri > tol:
                    solsM_with0 = np.column_stack((solsM_with0, concs_before))
                    statei, erri = self.find_state_match(solsM_with0, concs_before)

                if errj > tol:
                    solsM_with0 = np.column_stack((solsM_with0, concs_after))
                    statej, errj = self.find_state_match(solsM_with0, concs_after)

                if statei != statej:  # stop putting on the self-edges:
                    G_states.add_edge(statei, statej, transition=signal_labels[si])
                    if verbose:
                        print(f'From state {statei} with signal {signal_labels[si]} to state {statej}')

        self._signal_inds = signal_inds # save the list of signal inds
        self._signal_labels = signal_labels # save the list of node labels corresponding to the signals

        self._solsM = solsM_with0 # re-assign the solutions matrix with zero and any addition
        # states located during the creation of the steady-state network diagram.

        self.G_states = G_states # save the state transition diagram.

    def plot_transition_network_o(self,
                                save_graph_net: str):
        '''
        Plot the state transition diagram as a graphviz object,
        which unfortunately can't be directly displayed but is saved
        to disk as an image file.

        Parameters
        ----------
        save_graph_net : str
            The path and filename to save the network as an image file.

        '''

        edgedata_Gstates = nx.get_edge_attributes(self.G_states, "transition")
        nodes_Gstates = list(self.G_states.nodes)

        cmap = colormaps['rainbow_r']
        norm = colors.Normalize(vmin=0, vmax=self._solsM.shape[1])

        G = pgv.AGraph(strict=False,
                       splines=True,
                       directed=True,
                       concentrate=False,
                       dpi=300)

        for ni, nde_stateo in enumerate(nodes_Gstates):
            nde_color = colors.rgb2hex(cmap(norm(ni)))
            nde_color += '80'  # add some transparancy to the node
            nde_font_color = 'Black'

            nde_state = f'State {nde_stateo}'

            G.add_node(nde_state,
                       style='filled',
                       fillcolor=nde_color,
                       fontcolor=nde_font_color,
                       )

        for (eio, ejo, em), etranso in edgedata_Gstates.items():
            ei = f'State {eio}'
            ej = f'State {ejo}'
            if etranso == -1:
                etrans = 'Spont.'
            else:
                etrans = etranso
            G.add_edge(ei, ej, label=etrans)

        G.layout(prog='dot')  # default to dot

        G.draw(save_graph_net)