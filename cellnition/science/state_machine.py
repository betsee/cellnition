#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds and plots a state transition diagram from a solution
set and corresponding GeneNetworkModel.
'''
import itertools

#FIXME: allow for combination signals (i.e. s0*s1) to be studied (optionally) in
# the state transition diagram.

import numpy as np
from numpy import ndarray
from matplotlib import colors
from matplotlib import colormaps
import networkx as nx
from networkx import DiGraph
from cellnition.science.gene_networks import GeneNetworkModel

import pygraphviz as pgv

class StateMachine(object):
    '''
    Build and plots a state transition diagram from a solution set and
    corresponding GeneNetworkModel. This class uses time simulation,
    starting the system off at the zero vector plus every stable state in
    a supplied matrix, and by temporarily triggering signal nodes in the
    network, it then looks to see if there is a new stable state for the
    system after the perturbation. The transitions between states are
    recorded in a state transition diagram.

    Public Attributes
    -----------------
    G_states : DiGraph
        State transition network, showing how each steady-state of the
        network is reached through a signal transition.

    Private Attributes
    ------------------
    _gmod : GeneNetworkModel
        An instance of GeneNetworkModel

    _solsM : ndarray
        A set of steady-state solutions from _gmod.
    '''

    def __init__(self, gmod: GeneNetworkModel, solsM: ndarray):
        '''
        Initialize the StateMachine.

        Parameters
        ----------
        gmod : GeneNetworkModel
            An instance of GeneNetworkModel with an analytical model built.

        solsM : ndarray
            A set of unique steady-state solutions from the GeneNetworkModel.
            These will be the states of the StateMachine.
        '''
        self._gmod = gmod
        self._solsM = solsM

        self.G_states = None # The state transition network

    def find_state_match(self,
                         solsM: ndarray,
                         cvecti: list | ndarray) -> tuple[ndarray, float]:
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
                                  dt: float = 1.0e-3,
                                  tend: float = 100.0,
                                  sigtstart: float = 33.0,
                                  sigtend: float = 66.0,
                                  sigmax: float = 2.0,
                                  delta_window: float = 1.0,
                                  verbose: bool = True,
                                  tol: float = 1.0e-6,
                                  do_combos: bool=False
                                  ):
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

        sigtstart: float = 33.0
            The time to start the signal perturbation. Care must be taken
            to ensure enough time is alloted prior to starting the perturbation
            for the system to have reached an initial steady state.

        sigtend: float = 66.0
            The time to end the signal perturbation.

        sigmax: float = 2.0
            Magnitude to add to the signal nodes.

        delta_window: float = 1.0
            Time to sample prior to the application of the signal perturbation,
            in which the initial steady-state is collected.

        verbose: bool = True
            Print out log statements (True)?

        tol: float = 1.0e-6
            Tolerance, below which a state is accepted as a match. If the state
            match error is above tol, it is added to the matrix as a new state.

        do_combos : bool = False
            Consider combinations of all signal nodes in the network (e.g. s0*s1)
            as possible transitions for states.

        '''
        c_zeros = np.zeros(self._gmod.N_nodes)  # start everything out low

        # Create a steady-state solutions matrix that is stacked with the
        # 'zero' or 'base' state:
        solsM_with0 = np.column_stack((c_zeros, self._solsM))

        G_states = DiGraph()

        signal_inds = [[sigi] for sigi in self._gmod.signal_inds]

        if do_combos:
            extended_sigs = []
            for sigi in self._gmod.signal_inds:
                for sigj in self._gmod.signal_inds:
                    if sigi != sigj:
                        extended_sigs.append([sigi, sigj])
            extended_sigs.append(self._gmod.signal_inds) # append all of them as a list to do all at once
            signal_inds.extend(extended_sigs)

        # Make string labels for each of the signals:
        signal_labels = []
        for sigi in signal_inds:
            if len(sigi) > 1:
                str_lab = ''
                for i, nss in enumerate(sigi):
                    ss = self._gmod.nodes_list[nss]
                    if i == 0:
                        str_lab += f'{ss}'
                    else:
                        str_lab += f'*{ss}'
                signal_labels.append(str_lab)
            else:
                ss = self._gmod.nodes_list[sigi[0]]
                signal_labels.append(f'{ss}')

        for stateio, cvecto in enumerate(solsM_with0.T):  # start the system off in each of the states

            # For each signal in the network:
            for si, sigi in enumerate(signal_inds):
                cvecti = cvecto.copy()  # reset the state to the desired starting state
                sig_times = [(sigtstart, sigtend) for ss in sigi]
                sig_mags = [(0.0, sigmax) for ss in sigi]

                # Run the time simulation:
                concs_time, tvect = self._gmod.run_time_sim(tend,
                                                      dt,
                                                      cvecti,
                                                      sigi,
                                                      sig_times,
                                                      sig_mags,
                                                      dt_samp=0.15)

                it_30low = (tvect <= sigtstart - delta_window).nonzero()[0]
                it_30high = (tvect >= sigtstart - 2 * delta_window).nonzero()[0]
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

    def plot_transition_network(self,
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

        for (eio, ejo), etranso in edgedata_Gstates.items():
            ei = f'State {eio}'
            ej = f'State {ejo}'
            if etranso == -1:
                etrans = 'Spont.'
            else:
                etrans = etranso
            G.add_edge(ei, ej, label=etrans)

        G.layout(prog='dot')  # default to dot

        G.draw(save_graph_net)