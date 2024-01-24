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
from matplotlib import colors
from matplotlib import colormaps
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
                               Bi: float | list = 0.5,
                               ni: float | list = 3.0,
                               di: float | list = 1.0,
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

        # Create parameter vectors for the model:
        self._gmod.create_parameter_vects(Bi, ni, di)

        # Solve the system with all concentrations:
        sols_0 = self._gmod.optimized_phase_space_search(Ns=Ns,
                                                   cmax=cmax,
                                                   round_sol=round_sol,
                                                   Bi=self._gmod.B_vect,
                                                   di=self._gmod.d_vect,
                                                   ni=self._gmod.n_vect,
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

        for i, c_ko_s in enumerate(self._gmod.c_vect_s):  # Step through each concentration

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

            if self._gmod._include_process is False:
                lambda_params = [c_vect_ko,
                                 c_ko_s,
                                 self._gmod.d_vect_s,
                                 self._gmod.B_vect_s,
                                 self._gmod.n_vect_s,
                                 ]

            else:
                lambda_params = [c_vect_ko,
                                 c_ko_s,
                                 self._gmod.d_vect_s,
                                 self._gmod.B_vect_s,
                                 self._gmod.n_vect_s,
                                 self._gmod.process_params_s
                                 ]

            flatten_f = np.asarray([fs for fs in dcdt_vect_ko_s])
            dcdt_vect_ko_f = sp.lambdify(lambda_params, flatten_f)

            # knockout_dcdt_f_set.append(dcdt_vect_ko_f) # store for later use

            # Determine the set of additional arguments to the optimization function -- these are different each
            # time as the clamped concentration becomes an additional known parameter:
            if self._gmod._include_process is False:
                function_args = (0.0, self._gmod.d_vect, self._gmod.B_vect, self._gmod.n_vect)
            else:
                function_args = (0.0, self._gmod.d_vect, self._gmod.B_vect, self._gmod.n_vect,
                                 self._gmod.process_params_f)

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

                if self._gmod._include_process is False:  # If we're not using the process, constrain all concs to be above zero
                    if (np.all(np.asarray(sol_root) >= 0.0)):
                        mins_found.append(sol_root)
                else:
                    # get the nodes that must be constrained above zero:
                    conc_nodes = np.setdiff1d(self._gmod.nodes_index, self._gmod._process_i)
                    # Then, only the nodes that are gene products must be above zero
                    if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
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
                                 gmod: GeneNetworkModel,
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

        for i, c_ko_s in enumerate(gmod.c_vect_s):  # Step through each concentration

            c_vect_time = []
            cvecti = cvecti_o.copy()  # start all nodes off at the supplied initial conditions

            if i == 0:
                for ti, tt in enumerate(tvect):
                    # for the first run, do a wild-type simulation
                    dcdt = gmod.dcdt_vect_f(cvecti, gmod.d_vect, gmod.B_vect, gmod.n_vect)
                    cvecti += dt * dcdt

                    if tt in tvect_samp:
                        c_vect_time.append(cvecti.copy())

                ko_M.append(cvecti.copy())  # append the wt steady-state to knockout ss solutions array
                conc_vect_ko.append(c_vect_time)

            cvecti = cvecti_o.copy()  # reset nodes back to the supplied initial conditions
            c_vect_time = []

            if i in gmod.regular_node_inds:

                for ti, tt in enumerate(tvect):
                    dcdt = gmod.dcdt_vect_f(cvecti, gmod.d_vect, gmod.B_vect, gmod.n_vect)
                    cvecti += dt * dcdt

                    if ti > int(Nt / 2):
                        cvecti[i] = 0.0  # force the clamp down of the k.o. gene concentration

                    if tt in tvect_samp:
                        c_vect_time.append(cvecti.copy())

                ko_M.append(cvecti.copy()) # append the last solution set to the steady-state matrix
                conc_vect_ko.append(c_vect_time.copy())

        ko_M = np.asarray(ko_M).T
        conc_vect_ko = np.asarray(conc_vect_ko)

        return tvectr, conc_vect_ko, ko_M
    def gene_knockout_reduce_eq(self, verbose: bool = True):
        '''
        Performs a knockout of all genes in the network, attempting to analytically solve for the
        resulting knockout change vector at steady-state. This uses root-finding algorithms to
        solve the knockout system, so will find all steady-states.

        '''

        knockout_dcdt_reduced_s_set = []
        knockout_c_reduced_s_set = []
        knockout_sol_s_set = []  # for full analytical solution equations

        if self._gmod.dcdt_vect_s is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        for i, c_ko_s in enumerate(self._gmod.c_vect_s):  # Step through each concentration

            # Define a new change vector by substituting in the knockout value for the gene (c=0) and
            # clamping the gene at that level by setting its change rate to zero:
            dcdt_vect_ko_s = self._gmod.dcdt_vect_s.subs(c_ko_s, 0)
            dcdt_vect_ko_s[i] = 0

            nosol = False

            try:
                sol_csetoo = sp.nonlinsolve(dcdt_vect_ko_s, self._gmod.c_vect_s)
                # Clean up the sympy container for the solutions:
                sol_cseto = list(list(sol_csetoo)[0])

                if len(sol_cseto):

                    c_master_i = []  # the indices of concentrations involved in the master equations (the reduced dims)
                    sol_cset = {}  # A dictionary of auxillary solutions (plug and play)
                    for i, c_eq in enumerate(sol_cseto):
                        if c_eq in self._gmod.c_vect_s:  # If it's a non-solution for the term, append it as a non-reduced conc.
                            c_master_i.append(self._gmod.c_vect_s.index(c_eq))
                        else:  # Otherwise append the plug-and-play solution set:
                            sol_cset[self._gmod.c_vect_s[i]] = c_eq

                    sol_eqn_vect = []
                    for ci, eqi in sol_cset.items():
                        sol_eqn = sp.Eq(ci, eqi)
                        sol_eqn_vect.append(sol_eqn)

                    knockout_sol_s_set.append(sol_eqn_vect)  # append the expressions for the system solution

                    master_eq_list = []  # master equations to be numerically optimized (reduced dimension network equations)
                    c_vect_reduced = []  # concentrations involved in the master equations

                    if len(c_master_i):
                        for ii in c_master_i:
                            # substitute in the expressions in terms of master concentrations to form the master equations:
                            ci_solve_eq = dcdt_vect_ko_s[ii].subs([(k, v) for k, v in sol_cset.items()])
                            master_eq_list.append(ci_solve_eq)
                            c_vect_reduced.append(self._gmod.c_vect_s[ii])

                    else:  # if there's nothing in c_master_i but there are solutions in sol_cseto, then it's been fully solved:
                        if verbose:
                            print("Solution solved analytically!")

                    knockout_dcdt_reduced_s_set.append([])
                    knockout_c_reduced_s_set.append([])

                else:
                    nosol = True

            except:
                nosol = True

            if nosol:
                knockout_dcdt_reduced_s_set.append(dcdt_vect_ko_s)
                knockout_c_reduced_s_set.append(self._gmod.c_vect_s)

        return knockout_sol_s_set, knockout_dcdt_reduced_s_set, knockout_c_reduced_s_set

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