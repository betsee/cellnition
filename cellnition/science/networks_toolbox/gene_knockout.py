#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module implements a gene knockout experiment on a network model, where each
gene in the network is silenced and the new steady-states or dynamic activity is
determined.
'''
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from cellnition.science.network_models.probability_networks import ProbabilityNet
from scipy.cluster.hierarchy import fclusterdata

# FIXME: DOCUMENT THROUGHOUT

class GeneKnockout(object):
    '''
    Given a network model, this class contains routines to perform gene-knockout
    experiments (gene silencing) whereby individual genes are silenced and
    the behaviour of the network re-assessed.

    '''
    def __init__(self, pnet: ProbabilityNet):
        '''
        Initialize the class.

        Parameters
        ----------
        pnet : GeneNetworkModel
            An instance of GeneNetworkModel with an analytical model built;
            forms the basis of the knockout experiment.

        '''
        self._pnet = pnet # initialize the system

    def gene_knockout_ss_solve(self,
                               Ns: int = 3,
                               tol: float = 1.0e-15,
                               round_unique_sol: int = 2,
                               sol_tol: float = 1.0e-1,
                               d_base: float = 1.0,
                               n_base: float = 3.0,
                               beta_base: float = 4.0,
                               verbose: bool = True,
                               save_file_basename: str | None = None,
                               constraint_vals: list[float]|None = None,
                               constraint_inds: list[int]|None = None,
                               signal_constr_vals: list | None = None,
                               search_cycle_nodes_only: bool = False,
                               cluster_threshhold: float = 0.1,
                               cluster_method: str='distance'
                               ):
        '''
        Performs a sequential knockout of all genes in the network, computing all possible steady-state
        solutions for the resulting knockout. This is different from the transition matrix,
        as the knockouts aren't a temporary perturbation, but a long-term silencing.

        '''

        if constraint_vals is not None and constraint_inds is not None:
            if len(constraint_vals) != len(constraint_inds):
                raise Exception("Node constraint values must be same length as constrained node indices!")

        knockout_sol_set = [] # Master list to hold all -- possibly multistable -- solutions.
        knockout_header = [] # Header to hold an index of the gene knockout and the sub-solution indices

        if save_file_basename is not None:
            save_file_list = [f'{save_file_basename}_allc.csv']
            save_file_list.extend([f'{save_file_basename}_ko_c{i}.csv' for i in range(self._pnet.N_nodes)])

        else:
            save_file_list = [None]
            save_file_list.extend([None for i in range(self._pnet.N_nodes)])

        constrained_inds, constrained_vals = self._pnet._handle_constrained_nodes(constraint_inds,
                                                                                  constraint_vals)


        solsM, sol_M0_char, sols_0 = self._pnet.solve_probability_equms(constraint_inds=constrained_inds,
                                                                        constraint_vals=constrained_vals,
                                                                        signal_constr_vals=signal_constr_vals,
                                                                        d_base=d_base,
                                                                        n_base=n_base,
                                                                        beta_base=beta_base,
                                                                        N_space=Ns,
                                                                        search_tol=tol,
                                                                        sol_tol=sol_tol,
                                                                        N_round_sol=round_unique_sol,
                                                                        search_main_nodes_only=search_cycle_nodes_only
                                                                        )

        if verbose:
            print(f'-------------------')

        # print(f'size of solM before clustering: {solsM.shape}')

        # Cluster solutions to exclude those that are very similar
        solsM = self.find_unique_sols(solsM,
                                      cluster_threshhold=cluster_threshhold,
                                      cluster_method=cluster_method)

        # print(f'size of solM after clustering: {solsM.shape}')

        knockout_sol_set.append(solsM.copy()) # append the "wild-type" solution set
        knockout_header.extend([f'wt,' for j in range(solsM.shape[1])])

        for nde_i in self._pnet.nodes_index:

            if nde_i in self._pnet.input_node_inds:
                sig_ind = self._pnet.input_node_inds.index(nde_i)
                signal_constr_vals_mod = signal_constr_vals.copy()
                signal_constr_vals_mod[sig_ind] = self._pnet.p_min

                if constraint_vals is not None or constraint_inds is not None:
                    cvals = constraint_vals
                    cinds = constraint_inds
                else:
                    cvals = None
                    cinds = None

            else:
                signal_constr_vals_mod = signal_constr_vals

                if constraint_vals is None or constraint_inds is None:
                    # Gene knockout is the only constraint:
                    cvals = [self._pnet.p_min]
                    cinds = [nde_i]

                else: # add the gene knockout as a final constraint:
                    cvals = constraint_vals + [self._pnet.p_min]
                    cinds = constraint_inds + [nde_i]

            # We also need to add in naturally-occurring constraints from unregulated nodes:

            solsM, sol_M0_char, sols_1 = self._pnet.solve_probability_equms(constraint_inds=cinds,
                                                                            constraint_vals=cvals,
                                                                            signal_constr_vals=signal_constr_vals_mod,
                                                                            d_base=d_base,
                                                                            n_base=n_base,
                                                                            beta_base=beta_base,
                                                                            N_space=Ns,
                                                                            search_tol=tol,
                                                                            sol_tol=sol_tol,
                                                                            N_round_sol=round_unique_sol,
                                                                            verbose=verbose,
                                                                            search_main_nodes_only=search_cycle_nodes_only
                                                                            )

            if verbose:
                print(f'-------------------')

            # print(f'size of solM {i} before clustering: {solsM.shape}')

            # Cluster solutions to exclude those that are very similar
            solsM = self.find_unique_sols(solsM,
                                          cluster_threshhold=cluster_threshhold,
                                          cluster_method=cluster_method)

            # print(f'size of solM {i} after clustering: {solsM.shape}')

            knockout_sol_set.append(solsM.copy())
            knockout_header.extend([f'{self._pnet.nodes_list[nde_i]},' for j in range(solsM.shape[1])])

        # merge this into a master matrix:
        ko_M = None
        for i, ko_aro in enumerate(knockout_sol_set):
            if len(ko_aro) == 0:
                ko_ar = np.asarray([np.zeros(self._pnet.N_nodes)]).T
            else:
                ko_ar = ko_aro

            if i == 0:
                ko_M = ko_ar
            else:
                ko_M = np.hstack((ko_M, ko_ar))

        return knockout_sol_set, ko_M, knockout_header


    def plot_knockout_arrays(self, knockout_sol_set: list | ndarray, figsave: str=None):
            '''
            Plot all steady-state solution arrays in a knockout experiment solution set.

            '''

            # let's plot this as a multidimensional set of master arrays:
            knock_flat = []
            for kmat in knockout_sol_set:
                for ki in kmat:
                    knock_flat.extend(ki)

            vmax = np.max(knock_flat)
            vmin = np.min(knock_flat)

            cmap = 'magma'

            N_axis = len(knockout_sol_set)

            fig, axes = plt.subplots(1, N_axis, sharey=True, sharex=True)

            for i, (axi, solsMio) in enumerate(zip(axes, knockout_sol_set)):
                if len(solsMio):
                    solsMi = solsMio
                else:
                    solsMi = np.asarray([np.zeros(self._pnet.N_nodes)]).T
                axi.imshow(solsMi, aspect="equal", vmax=vmax, vmin=vmin, cmap=cmap)
                axi.axis('off')
                if i != 0:
                    axi.set_title(f'c{i - 1}')
                else:
                    axi.set_title(f'Full')

            if figsave is not None:
                plt.savefig(figsave, dpi=300, transparent=True, format='png')

            return fig, axes

    def find_unique_sols(self,
                         solsM,
                         cluster_threshhold: float=0.1,
                         cluster_method: str='distance',
                         N_round_sol: int=2):
        '''

        '''

        if solsM.shape[1] > 1:
            unique_sol_clusters = fclusterdata(solsM.T, t=cluster_threshhold, criterion=cluster_method)

            cluster_index = np.unique(unique_sol_clusters)

            cluster_pool = [[] for i in cluster_index]
            for i, clst_i in enumerate(unique_sol_clusters):
                cluster_pool[int(clst_i) - 1].append(i)

            solsM_all_unique = np.zeros((self._pnet.N_nodes, len(cluster_pool)))

            for ii, sol_i in enumerate(cluster_pool):
                if len(sol_i):
                    solsM_all_unique[:, ii] = (np.mean(solsM[:, sol_i], 1))

            # redefine the solsM data structures:
            solsM = solsM_all_unique

            # # # first use numpy unique on rounded set of solutions to exclude similar cases:
            _, inds_solsM_all_unique = np.unique(np.round(solsM, N_round_sol), return_index=True, axis=1)
            solsM = solsM[:, inds_solsM_all_unique]

        return solsM
