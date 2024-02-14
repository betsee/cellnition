#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
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
                               search_cycle_nodes_only: bool = False
                               ):
        '''
        Performs a sequential knockout of all genes in the network, computing all possible steady-state
        solutions for the resulting knockout. This is different from the transition matrix,
        as the knockouts aren't a temporary perturbation, but a long-term silencing.

        '''

        if constraint_vals is not None and constraint_inds is not None:
            if len(constraint_vals) != len(constraint_inds):
                raise Exception("Node constraint values must be same length as constrained node indices!")

        knockout_sol_set = []

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
                                                                        search_cycle_nodes_only=search_cycle_nodes_only
                                                                        )

        if verbose:
            print(f'-------------------')

        knockout_sol_set.append(solsM.copy()) # append the "wild-type" solution set

        for i in self._pnet.regular_node_inds:  # Include only 'gene' nodes as silenced

            if constraint_vals is None or constraint_inds is None:
                # Gene knockout is the only constraint:
                cvals = [0.0]
                cinds = [i]

            else: # add the gene knockout as a final constraint:
                cvals = constraint_vals + [0.0]
                cinds = constraint_inds + [i]

            # We also need to add in naturally-occurring constraints from unregulated nodes:

            solsM, sol_M0_char, sols_1 = self._pnet.solve_probability_equms(constraint_inds=cinds,
                                                                        constraint_vals=cvals,
                                                                        signal_constr_vals=signal_constr_vals,
                                                                        d_base=d_base,
                                                                        n_base=n_base,
                                                                        beta_base=beta_base,
                                                                        N_space=Ns,
                                                                        search_tol=tol,
                                                                        sol_tol=sol_tol,
                                                                        N_round_sol=round_unique_sol,
                                                                        verbose=verbose,
                                                                        search_cycle_nodes_only=search_cycle_nodes_only
                                                                            )

            if verbose:
                print(f'-------------------')

            knockout_sol_set.append(solsM.copy())

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

        return knockout_sol_set, ko_M

    # def gene_knockout_time_solve(self,
    #                              tend: float,
    #                              dt: float,
    #                              cvecti_o: ndarray|list,
    #                              dt_samp: float|None = None,
    #                              d_base: float = 1.0,
    #                              n_base: float = 3.0,
    #                              beta_base: float = 4.0,
    #                              constraint_vals: list[float] | None = None,
    #                              constraint_inds: list[int] | None = None,
    #                              ):
    #     '''
    #
    #     '''
    #
    #     # Let's try knockouts as a time-sim thing:
    #     Nt = int(tend / dt)
    #
    #     tvect = np.linspace(0.0, tend, Nt)
    #
    #     conc_timevect_ko = []
    #     knockout_sol_set = [] # matrix storing solutions at steady state
    #
    #     # make a time-step update vector so we can update any sensors as
    #     # an absolute reading (dt = 1.0) while treating the kinetics of the
    #     # other node types:
    #     dtv = 1.0e-3 * np.ones(self._pnet.N_nodes)
    #     dtv[self._pnet.sensor_node_inds] = 1.0
    #
    #     wild_type_sim = False
    #     for i in self._pnet.regular_node_inds:  # Step through each regular gene index
    #         cvecti = cvecti_o.copy()  # start all nodes off at the supplied initial conditions
    #
    #         if wild_type_sim is False: # perform a wild-type simulation
    #             c_vect_time, tvectr = self._pnet.run_time_sim(tend,
    #                                                      dt,
    #                                                      cvecti,
    #                                                      sig_inds=None,
    #                                                      sig_times=None,
    #                                                      sig_mag=None,
    #                                                      dt_samp=dt_samp,
    #                                                      constrained_inds=constraint_inds,
    #                                                      constrained_vals=constraint_vals,
    #                                                      d_base=d_base,
    #                                                      n_base=n_base,
    #                                                      beta_base=beta_base
    #                                                      )
    #
    #             knockout_sol_set.append(1*c_vect_time[-1])  # append the wt steady-state to knockout ss solutions array
    #             conc_timevect_ko.append(c_vect_time)
    #             wild_type_sim = True # set to true so it's not done again
    #
    #         cvecti = cvecti_o.copy()  # reset nodes back to the supplied initial conditions
    #
    #         # Now alter node values as signals
    #         tsig = [(tvect[int(Nt/2)], 2*tend)]  # start and end time for the silencing
    #         isig = [i] # index for silencing
    #         magsig = [(1.0, 0.0)] # magnitude of signal switch before and after
    #
    #         c_vect_time, tvectr = self._pnet.run_time_sim(tend,
    #                                                  dt,
    #                                                  cvecti,
    #                                                  sig_inds=tsig,
    #                                                  sig_times=isig,
    #                                                  sig_mag=magsig,
    #                                                  dt_samp=dt_samp,
    #                                                  constrained_inds=constraint_inds,
    #                                                  constrained_vals=constraint_vals,
    #                                                  d_base=d_base,
    #                                                  n_base=n_base,
    #                                                  beta_base=beta_base
    #                                                  )
    #
    #         knockout_sol_set.append(1 * c_vect_time[-1])  # append the wt steady-state to knockout ss solutions array
    #         conc_timevect_ko.append(c_vect_time)
    #
    #     knockout_sol_set = np.asarray(knockout_sol_set).T
    #     conc_timevect_ko = np.asarray(conc_timevect_ko)
    #
    #     return tvectr, conc_timevect_ko, knockout_sol_set

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