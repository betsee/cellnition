'''
This module has methods to search the state space or the parameter space of the model
for desired attributes.
'''
import numpy as np
from numpy import ndarray
from cellnition.science.gene_networks import GeneNetworkModel


def multistability_search(gmod: GeneNetworkModel,
                          N_multi: int = 1,
                          tol: float = 1.0e-3,
                          N_iter: int = 100,
                          verbose: bool = True,
                          N_space: int = 3,
                          N_round_sol: int = 6,
                          N_round_unique_sol: int = 1,
                          Ki: float | list = 0.5,
                          ni: float | list = 3.0,
                          di: float | list = 1.0,
                          search_tol: float = 1.0e-15,
                          add_interactions: bool = True,
                          unique_sols: bool = True
                          ) -> tuple[list, list]:
    '''
    By randomly generating sets of different edge interaction types (i.e. activator or inhibitor), find
    as many unique multistable systems as possible for a given base network.

    Parameters
    ----------
    gmod: GeneNetworkModel
        An instance of the GeneNetworkModel with an analytical model built.

    N_multi : int
        The solutions with N_multi minimum number of stable states will be added to the set.

    N_iter : int = 100
        The number of times edge_types should be randomly generated and simulated.

    N_space: int=3
        The number of points to consider along each axis of the state space search.

    Ki: float|list = 0.5
        The Hill coefficient.

    ni: float|list = 3.0
        The Hill exponent.

    di: float|list = 0.5
        If hold_d_const, this is the value to hold the maximum decay rate at.

    N_round_unique_sol: int = 1
        Digit to round solutions to prior to determining uniqueness.

    search_round_sol: int=6
        The number of digits to round solutions to in state space search.

    tol: float=1.0e-3
        The tolerance below which solutions are considered robust enough to
        include in the solution set.

    cmax_multi: float=2.0
        The maximum concentration value to search for in the state space search,
        where this is also multiplied by the maximum in-degree of the network.

    verbose: bool=True
        Output print messages (True)?

    search_tol : float = 1.0e-15
        The tolerance to search in the root-finding algorithm.

    add_interactions : bool = True
        For nodes with two or more interactions, do these add (True) or multiply (False)?

    unique_sols : bool = True
        Record only unique steady-state solutions (True)?

    Returns
    -------
    numsol_list : list
        A list of the number of solutions returned for each successful search.

    multisols : list
        A list of the solution set and the edge types for each successful search.

    '''

    multisols = []
    multisol_edges = []
    numsol_list = []

    for i in range(N_iter):
        edge_types = gmod.get_edge_types(p_acti=0.5)
        gmod.build_analytical_model(edge_types=edge_types,
                                    add_interactions=add_interactions)
        sols_0 = gmod.optimized_phase_space_search(Ns=N_space,
                                                   cmax=1.5 * np.max(gmod.in_degree_sequence),
                                                   round_sol=N_round_sol,
                                                   Ki=Ki,
                                                   di=di,
                                                   ni=ni,
                                                   tol=search_tol,
                                                   method="Root"
                                                   )

        solsM = gmod.find_attractor_sols(sols_0,
                                         tol=tol,
                                         unique_sols=unique_sols,
                                         verbose=False,
                                         N_round=N_round_unique_sol)

        if len(solsM):
            num_sols = solsM.shape[1]
        else:
            num_sols = 0

        if num_sols >= N_multi:
            edge_types_l = edge_types.tolist()
            if edge_types_l not in multisol_edges:  # If we don't already have this combo:
                if verbose:
                    print(f'Found solution with {num_sols} states on iteration {i}')
                multisols.append([sols_0, edge_types])
                numsol_list.append(num_sols)
                multisol_edges.append(edge_types_l)

    return numsol_list, multisols


def param_space_search(gmod: GeneNetworkModel,
                       N_pts: int=3,
                       ri: float = 1.0,
                       ni: float = 3.0,
                       K_min: float = 0.1,
                       K_max: float = 2.0,
                       d_min: float = 0.5,
                       d_max: float = 10.0,
                       sol_round: int = 1,
                       N_search: int = 3,
                       search_round_sol: int=6,
                       tol: float=1.0e-3,
                       cmax_multi: float=2.0,
                       verbose: bool=True,
                       hold_d_const: bool=True,
                       di: float=0.5) -> tuple[ndarray, list]:
    '''
    Search parameter space of a model to find parameter combinations that give different multistable
    states. This search only looks for changes in the Hill coefficient and the maximum decay rate,
    holding the Hill constant and the maximum growth rate parameters constant.

    Parameters
    ----------
    gmod: GeneNetworkModel
        An instance of the GeneNetworkModel with an analytical model built.

    N_pts: int=3
        The number of points to consider along each axis of the parameter space.

    ri: float = 1.0
        The maximum growth rate.

    ni: float = 3.0
        The Hill exponent.

    K_min: float = 0.1
        The minimum value for the Hill coefficient of each reaction.

    K_max: float = 2.0
        The maximum value for the Hill coefficient of each reaction.

    d_min: float = 0.5
        The minimum value for the max decay of each reaction.

    d_max: float = 10.0
        The maximum value for the max decay of each reaction.

    sol_round: int = 1
        Digit to round solutions to prior to determining uniqueness.

    N_search: int = 3
        The number of points to search in state space axis.

    search_round_sol: int=6
        The number of digits to round solutions to in state space search.

    tol: float=1.0e-3
        The tolerance below which solutions are considered robust enough to
        include in the solution set.

    cmax_multi: float=2.0
        The maximum concentration value to search for in the state space search,
        where this is also multiplied by the maximum in-degree of the network.

    verbose: bool=True
        Output print messages (True)?

    hold_d_const: bool=True
        Hold the maximum rate of decay constant and only optimize for the Hill
        coefficient (True)?

    di: float=0.5
        If hold_d_const, this is the value to hold the maximum decay rate at.

    Returns
    -------
    bif_space_M : ndarray
        An array that has all Hill coefficients (K-constants), Hill decay constants, and the
        number of steady-state solutions packed into each row of the array.

    sols_space_M : list
        An array that has all steady-state solutions stacked into the list.

    '''

    if gmod._reduced_dims and gmod._solved_analytically is False:
        N_nodes = len(gmod.c_vect_reduced_s)

    else:
        N_nodes = gmod.N_nodes

    # What we wish to create is a parameter space search, as this net is small enough to enable that.
    Klin = np.linspace(K_min, K_max, N_pts)
    dlin = np.linspace(d_min, d_max, N_pts)

    param_lin_set = []

    for edj_i in range(gmod.N_edges):
        param_lin_set.append(Klin*1) # append the linear K-vector choices for each edge

    if hold_d_const is False:
        for nde_i in range(N_nodes):
            param_lin_set.append(dlin*1)

    # Create a set of matrices specifying the concentration grid for each
    # node of the network:
    param_M_SET = np.meshgrid(*param_lin_set, indexing='ij')

    # Create linearized arrays for each concentration, stacked into one column per node:
    param_test_set = np.asarray([pM.ravel() for pM in param_M_SET]).T

    bif_space_M = [] # Matrix holding the parameter values and number of unique stable solutions
    sols_space_M = []

    if verbose:
        print(param_M_SET[0].ravel().shape)

    if verbose:
        print(f'Search cmax will be {cmax_multi * np.max(gmod.in_degree_sequence)}')

    for param_set_i in param_test_set:
        kvecti = param_set_i[0:gmod.N_edges].tolist()

        if hold_d_const is False:
            dvecti = param_set_i[gmod.N_edges:].tolist()
        else:
            dvecti = di

        gmod.create_parameter_vects(Ki=kvecti, ni=ni, ri=ri, di=dvecti)

        sols_0 = gmod.optimized_phase_space_search(Ns=N_search,
                                                   cmax=cmax_multi * np.max(gmod.in_degree_sequence),
                                                   round_sol=search_round_sol,
                                                   Ki=gmod.K_vect,
                                                   di=gmod.d_vect,
                                                   ni=gmod.n_vect,
                                                   tol=1.0e-6,
                                                   method="Root"
                                                   )

        solsM = gmod.find_attractor_sols(sols_0, tol=tol, verbose=False, unique_sols=True, sol_round=sol_round)

        if len(solsM):
            num_sols = solsM.shape[1]
        else:
            num_sols = 0

        bif_space_M.append([*gmod.K_vect, *gmod.d_vect, num_sols])
        sols_space_M.append(solsM)

    return np.asarray(bif_space_M), sols_space_M