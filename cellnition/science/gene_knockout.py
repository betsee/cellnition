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
# FIXME: Implement state-grabbing from the time-sim

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
                               tol: float = 1.0e-6,
                               round_sol: int = 6,
                               round_unique_sol: int = 2,
                               unique_sols: bool = True,
                               sol_tol: float = 1.0e-3,
                               verbose: bool = True,
                               save_file_basename: str | None = None
                               ):
        '''
        Performs a sequential knockout of all genes in the network, computing all possible steady-state
        solutions for the resulting knockout. This is different from the transition matrix,
        as the knockouts aren't a temporary perturbation, but a long-term silencing.

        '''

        if self._gmod.dcdt_vect_s is None:
            raise Exception("Must use the method build_analytical_model in GeneNetworkModel "
                            "to generate attributes required to run gene knockout sims.")

        knockout_sol_set = []
        # knockout_dcdt_s_set = []
        # knockout_dcdt_f_set = []

        if save_file_basename is not None:
            save_file_list = [f'{save_file_basename}_allc.csv']
            save_file_list.extend([f'{save_file_basename}_ko_c{i}.csv' for i in range(self._gmod.N_nodes)])

        else:
            save_file_list = [None]
            save_file_list.extend([None for i in range(self._gmod.N_nodes)])

        # Solve the system with all concentrations:
        sols_0 = self._gmod.optimized_phase_space_search(Ns=Ns,
                                                   cmax=cmax,
                                                   round_sol=round_sol,
                                                   tol=tol,
                                                   method="Root"
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

        for i in self._gmod.regular_node_inds:  # Include only 'gene' nodes

            c_ko_s = self._gmod.c_vect_s[i]  # silence only genes
            # Define a new change vector by substituting in the knockout value for the gene (c=0) and
            # clamping the gene at that level by setting its change rate to zero:
            dcdt_vect_ko_s = self._gmod.dcdt_vect_s.copy()  # make a copy of the symbolic change vector
            # dcdt_vect_ko_s = dcdt_vect_ko_s.subs(c_ko_s, 0)
            dcdt_vect_ko_s.row_del(i)  # Now we have to remove the row for this concentration

            # create a new symbolic concentration vector that has the silenced gene removed:
            c_vect_ko = self._gmod.c_vect_s.copy()
            c_vect_ko.remove(c_ko_s)

            # do a similar thing for conc. indices so we can reconstruct solutions easily:
            nodes_ko = self._gmod.nodes_index.copy()
            del nodes_ko[i]  # delete the ith index

            # knockout_dcdt_s_set.append(dcdt_vect_ko_s) # Store for later
            lambda_params = self._gmod._fetch_lambda_params_s(c_vect_ko, c_ko_s)
            function_args = self._gmod._fetch_function_args_f(0.0)

            flatten_f = np.asarray([fs for fs in dcdt_vect_ko_s])
            dcdt_vect_ko_f = sp.lambdify(lambda_params, flatten_f)

            # knockout_dcdt_f_set.append(dcdt_vect_ko_f) # store for later use

            # Determine the set of additional arguments to the optimization function -- these are different each
            # time as the clamped concentration becomes an additional known parameter:

            # Generate the points in state space to sample at:
            c_test_set, _, _ = self._gmod.generate_state_space(c_vect_ko,
                                                               Ns=Ns,
                                                               cmin=cmin,
                                                               cmax=cmax,
                                                               include_signals=True)

            # Initialize the equillibrium point solutions to be a set:
            mins_found = []

            for c_vecti in c_test_set:

                sol_rooto = fsolve(dcdt_vect_ko_f, c_vecti, args=function_args, xtol=tol)

                # reconstruct a full-length concentration vector:
                sol_root = np.zeros(self._gmod.N_nodes)
                # the solution is defined at the remaining nodes; the unspecified value is the silenced gene
                sol_root[nodes_ko] = sol_rooto

                if len(self._gmod.process_node_inds):
                    # get the nodes that must be constrained above zero:
                    conc_nodes = np.setdiff1d(self._gmod.nodes_index, self._gmod.process_node_inds)
                    # Then, only the nodes that are gene products must be above zero
                    if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
                        mins_found.append(sol_root)

                else:  # If we're not using the process, constrain all concs to be above zero
                    if (np.all(np.asarray(sol_root) >= 0.0)):
                        mins_found.append(sol_root)

                mins_found = np.round(mins_found, round_sol)
                mins_found = np.unique(mins_found, axis=0).tolist()

            if verbose:
                print(f'Steady-state solutions for {self._gmod.c_vect_s[i].name} knockout:')

            # Screen only for attractor solutions:
            solsM = self._gmod.find_attractor_sols(mins_found,
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