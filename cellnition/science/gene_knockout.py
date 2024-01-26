#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module implements a gene knockout experiment on a network model, where each
gene in the network is silenced and the new steady-states or dynamic activity is
determined.
'''
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve
import sympy as sp
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
from cellnition.science.gene_networks import GeneNetworkModel

# FIXME: DOCUMENT THROUGHOUT

class GeneKnockout(object):
    '''
    Given a network model, this class contains routines to perform gene-knockout
    experiments (gene silencing) whereby individual genes are silenced and
    the behaviour of the network re-assessed.

    '''
    def __init__(self, gmod: GeneNetworkModel):
        '''
        Initialize the class.

        Parameters
        ----------
        gmod : GeneNetworkModel
            An instance of GeneNetworkModel with an analytical model built;
            forms the basis of the knockout experiment.

        '''
        self._gmod = gmod # initialize the system

    def gene_knockout_ss_solve(self,
                               Ns: int = 3,
                               cmin: float = 0.0,
                               cmax: float = 1.0,
                               tol: float = 1.0e-15,
                               round_sol: int = 6,
                               round_unique_sol: int = 2,
                               unique_sols: bool = True,
                               sol_tol: float = 1.0e-1,
                               verbose: bool = True,
                               save_file_basename: str | None = None,
                               constraint_vals: list[float]|None = None,
                               constraint_inds: list[int]|None = None,
                               solver_method: str = 'Root'
                               ):
        '''
        Performs a sequential knockout of all genes in the network, computing all possible steady-state
        solutions for the resulting knockout. This is different from the transition matrix,
        as the knockouts aren't a temporary perturbation, but a long-term silencing.

        '''
        print("Knockout Experiments-----")

        if self._gmod.dcdt_vect_s is None:
            raise Exception("Must use the method build_analytical_model in GeneNetworkModel "
                            "to generate attributes required to run gene knockout sims.")

        if constraint_vals is not None and constraint_inds is not None:
            if len(constraint_vals) != len(constraint_inds):
                raise Exception("Node constraint values must be same length as constrained node indices!")

        knockout_sol_set = []

        if save_file_basename is not None:
            save_file_list = [f'{save_file_basename}_allc.csv']
            save_file_list.extend([f'{save_file_basename}_ko_c{i}.csv' for i in range(self._gmod.N_nodes)])

        else:
            save_file_list = [None]
            save_file_list.extend([None for i in range(self._gmod.N_nodes)])

        if constraint_vals is None or constraint_inds is None:
            # Solve the system with all concentrations:
            sols_0 = self._gmod.optimized_phase_space_search(Ns=Ns,
                                                             cmin=cmin,
                                                             cmax=cmax,
                                                             round_sol=round_sol,
                                                             tol=tol,
                                                             method=solver_method
                                                             )

        else:
            sols_0 = self._gmod.constrained_phase_space_search(constraint_vals,
                                                               constraint_inds,
                                                               Ns=Ns,
                                                               cmin=cmin,
                                                               cmax=cmax,
                                                               tol=tol,
                                                               round_sol=round_sol,
                                                               method=solver_method
                                                               )

        # Screen only for attractor solutions:
        solsM = self._gmod.find_attractor_sols(sols_0,
                                         tol=sol_tol,
                                         verbose=verbose,
                                         unique_sols=unique_sols,
                                         sol_round=round_unique_sol,
                                         save_file=save_file_list[0])

        if verbose:
            print(f'-------------------')

        knockout_sol_set.append(solsM.copy()) # append the "wild-type" solution set

        for i in self._gmod.regular_node_inds:  # Include only 'gene' nodes as silenced

            if constraint_vals is None or constraint_inds is None:
                # Gene knockout is the only constraint:
                cvals = [0.0]
                cinds = [i]

            else: # add the gene knockout as a final constraint:
                cvals = constraint_vals + [0.0]
                cinds = constraint_inds + [i]

            sols_1 = self._gmod.constrained_phase_space_search(cvals,
                                                               cinds,
                                                               Ns=Ns,
                                                               cmin=cmin,
                                                               cmax=cmax,
                                                               tol=tol,
                                                               round_sol=round_sol,
                                                               method=solver_method
                                                               )

            # Screen only for attractor solutions:
            solsM = self._gmod.find_attractor_sols(sols_1,
                                             tol=sol_tol,
                                             verbose=verbose,
                                             unique_sols=unique_sols,
                                             sol_round=round_unique_sol,
                                             save_file=save_file_list[i + 1])

            if verbose:
                print(f'-------------------')

            knockout_sol_set.append(solsM.copy())

        # merge this into a master matrix:
        ko_M = None
        for i, ko_aro in enumerate(knockout_sol_set):
            if len(ko_aro) == 0:
                ko_ar = np.asarray([np.zeros(self._gmod.N_nodes)]).T
            else:
                ko_ar = ko_aro

            if i == 0:
                ko_M = ko_ar
            else:
                ko_M = np.hstack((ko_M, ko_ar))

        return knockout_sol_set, ko_M

    def gene_knockout_time_solve(self,
                                 tend: float,
                                 dt: float,
                                 cvecti_o: ndarray|list,
                                 dt_samp: float|None = None):
        '''

        '''

        # Let's try knockouts as a time-sim thing:
        Nt = int(tend / dt)

        tvect = np.linspace(0.0, tend, Nt)

        # sampling compression
        if dt_samp is not None:
            sampr = int(dt_samp / dt)
            tvect_samp = tvect[0::sampr]
            tvectr = tvect_samp
        else:
            tvect_samp = None
            tvectr = tvect

        conc_vect_ko = []
        ko_M = [] # matrix storing solutions at steady state

        # make a time-step update vector so we can update any sensors as
        # an absolute reading (dt = 1.0) while treating the kinetics of the
        # other node types:
        dtv = 1.0e-3 * np.ones(self._gmod.N_nodes)
        dtv[self._gmod.sensor_node_inds] = 1.0

        function_args = self._gmod._fetch_function_args_f()
        wild_type_sim = False
        for i in self._gmod.regular_node_inds:  # Step through each regular gene index
            c_vect_time = []
            cvecti = cvecti_o.copy()  # start all nodes off at the supplied initial conditions

            if wild_type_sim is False: # perform a wild-type simulation
                for ti, tt in enumerate(tvect):
                    # for the first run, do a wild-type simulation
                    dcdt = self._gmod.dcdt_vect_f(cvecti, *function_args)
                    cvecti += dtv * dcdt

                    if tt in tvect_samp:
                        c_vect_time.append(cvecti.copy())

                ko_M.append(cvecti.copy())  # append the wt steady-state to knockout ss solutions array
                conc_vect_ko.append(c_vect_time)
                wild_type_sim = True # set to true so it's not done again

            cvecti = cvecti_o.copy()  # reset nodes back to the supplied initial conditions
            c_vect_time = []

            for ti, tt in enumerate(tvect):
                dcdt = self._gmod.dcdt_vect_f(cvecti, *function_args)
                cvecti += dtv * dcdt

                if ti > int(Nt / 2):
                    cvecti[i] = 0.0  # force the silencing of the k.o. gene concentration

                if tt in tvect_samp:
                    c_vect_time.append(cvecti.copy())

                ko_M.append(cvecti.copy()) # append the last solution set to the steady-state matrix
                conc_vect_ko.append(c_vect_time.copy())

        ko_M = np.asarray(ko_M).T
        conc_vect_ko = np.asarray(conc_vect_ko)

        return tvectr, conc_vect_ko, ko_M

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
                    solsMi = np.asarray([np.zeros(self._gmod.N_nodes)]).T
                axi.imshow(solsMi, aspect="equal", vmax=vmax, vmin=vmin, cmap=cmap)
                axi.axis('off')
                if i != 0:
                    axi.set_title(f'c{i - 1}')
                else:
                    axi.set_title(f'Full')

            if figsave is not None:
                plt.savefig(figsave, dpi=300, transparent=True, format='png')

            return fig, axes