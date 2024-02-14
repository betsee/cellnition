'''
This module has methods to search the state space or the parameter space of the model
for desired attributes.
'''
import numpy as np
from numpy import ndarray
from cellnition.science.network_models.probability_networks import ProbabilityNet
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             InterFuncType,
                                                             CouplingType)


# FIXME: we'd like to remove signal node edges and signal nodes from this search.

def multistability_search(pnet: ProbabilityNet,
                          N_multi: int = 1,
                          sol_tol: float = 1.0e-1,
                          N_iter: int = 5,
                          verbose: bool = True,
                          beta_base: float | list = 2.0,
                          n_base: float | list = 3.0,
                          d_base: float | list = 1.0,
                          N_space: int = 2,
                          N_round_unique_sol: int = 1,
                          search_tol: float = 1.0e-15,
                          constraint_vals: list[float]|None = None,
                          constraint_inds: list[int]|None = None,
                          signal_constr_vals: list|None = None,
                          coupling_type: CouplingType = CouplingType.mixed,
                          search_cycle_nodes_only: bool=False
                          ) -> tuple[list, list]:
    '''
    By randomly generating sets of different edge interaction types (i.e. activator or inhibitor), find
    as many unique multistable systems as possible for a given base network.

    Parameters
    ----------
    pnet: GeneNetworkModel
        An instance of the GeneNetworkModel with an analytical model built.

    N_multi : int
        The solutions with N_multi minimum number of stable states will be added to the set.

    N_iter : int = 100
        The number of times edge_types should be randomly generated and simulated.

    N_space: int=3
        The number of points to consider along each axis of the state space search.

    N_round_unique_sol: int = 1
        Digit to round solutions to prior to determining uniqueness.

    search_round_sol: int=6
        The number of digits to round solutions to in state space search.

    sol_tol: float=1.0e-3
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

    constraint_vals : list[float]|None = None
        Values for nodes that are to be constrained in the optimization problem. Must be
        same length as constraint_inds. If either constraint_vals or constraint_inds are
        None neither will be used.

    constraint_int : list[int]|None = None
        Indices of nodes that are to be constrained in the optimization problem. Must be
        same length as constraint_vals. If either constraint_vals or constraint_inds are
        None neither will be used.

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

    if constraint_vals is not None and constraint_inds is not None:
        if len(constraint_vals) != len(constraint_inds):
            raise Exception("Node constraint values must be same length as constrained node indices!")

    for i in range(N_iter):
        edge_types = pnet.get_edge_types(p_acti=0.5)

        # set the edge and node types to the network:
        pnet.set_edge_types(edge_types)

        # Get the adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                                              coupling_type=coupling_type)
        # build the analytical model for this network:
        pnet.build_analytical_model(A_add_s, A_mul_s)

        solsM, sol_M0_char, sol_0 = pnet.solve_probability_equms(constraint_inds=constraint_inds,
                                                                 constraint_vals=constraint_vals,
                                                                 signal_constr_vals=signal_constr_vals,
                                                                 d_base=d_base,
                                                                 n_base=n_base,
                                                                 beta_base=beta_base,
                                                                 N_space=N_space,
                                                                 search_tol=search_tol,
                                                                 sol_tol=sol_tol,
                                                                 N_round_sol=N_round_unique_sol,
                                                                 save_file=None,
                                                                 verbose=verbose,
                                                                 search_cycle_nodes_only=search_cycle_nodes_only
                                                                 )

        if len(solsM):
            num_sols = solsM.shape[1]
        else:
            num_sols = 0

        if num_sols >= N_multi:
            edge_types_l = edge_types.tolist()
            if edge_types_l not in multisol_edges:  # If we don't already have this combo:
                if verbose:
                    print(f'Found solution with {num_sols} states on iteration {i}')
                multisols.append([sol_0, edge_types])
                numsol_list.append(num_sols)
                multisol_edges.append(edge_types_l)

    return numsol_list, multisols


def param_space_search(pnet: ProbabilityNet,
                       N_pts: int=3,
                       n_base: float | list = 3.0,
                       beta_min: float = 2.0,
                       beta_max: float = 10.0,
                       N_unique_sol_round: int = 1,
                       N_search: int = 2,
                       sol_tol: float=1.0e-3,
                       search_tol: float=1.0e-3,
                       verbose: bool=True,
                       constraint_vals: list[float] | None = None,
                       constraint_inds: list[int] | None = None,
                       signal_constr_vals: list|None = None,
                       search_cycle_nodes_only: bool=False
                       ) -> tuple[ndarray, list]:
    '''
    Search parameter space of a model to find parameter combinations that give different multistable
    states. This search only looks for changes in the Hill coefficient and the maximum decay rate,
    holding the Hill constant and the maximum growth rate parameters constant.

    Parameters
    ----------
    pnet: GeneNetworkModel
        An instance of the GeneNetworkModel with an analytical model built.

    N_pts: int=3
        The number of points to consider along each axis of the parameter space.

    n_base: float|list = 3.0
        The Hill exponent (held constant).

    beta_min: float = 0.1
        The minimum value for the beta coefficient of each interaction edge.

    beta_max: float = 2.0
        The maximum value for the beta coefficient of each interaction edge.

    N_unique_sol_round: int = 1
        Digit to round solutions to prior to determining uniqueness.

    N_search: int = 3
        The number of points to search in state space axis.

    search_round_sol: int=6
        The number of digits to round solutions to in state space search.

    sol_tol: float=1.0e-3
        The tolerance below which solutions are considered robust enough to
        include in the solution set.

    cmax_multi: float=2.0
        The maximum concentration value to search for in the state space search,
        where this is also multiplied by the maximum in-degree of the network.

    verbose: bool=True
        Output print messages (True)?

    coi: float|list = 0.0
        The centre of any sensor's logistic functions (held constant).

    ki: float|list = 10.0
        The rate of rise of any sensor's logistic functions (held constant).

    constraint_vals : list[float]|None = None
        Values for nodes that are to be constrained in the optimization problem. Must be
        same length as constraint_inds. If either constraint_vals or constraint_inds are
        None neither will be used.

    constraint_int : list[int]|None = None
        Indices of nodes that are to be constrained in the optimization problem. Must be
        same length as constraint_vals. If either constraint_vals or constraint_inds are
        None neither will be used.

    Returns
    -------
    bif_space_M : ndarray
        An array that has all beta coefficients and the
        number of steady-state solutions packed into each row of the array.

    sols_space_M : list
        An array that has all steady-state solutions stacked into the list.

    '''

    if constraint_vals is not None and constraint_inds is not None:
        if len(constraint_vals) != len(constraint_inds):
            raise Exception("Node constraint values must be same length as constrained node indices!")

    # What we wish to create is a parameter space search, as this net is small enough to enable that.
    beta_lin = np.linspace(beta_min, beta_max, N_pts)

    beta_lin_set = []

    for edj_i in range(pnet.N_edges):
        beta_lin_set.append(beta_lin*1) # append the beta-vector choices for each edge

    # Create a set of matrices specifying the concentration grid for each
    # node of the network:
    beta_M_SET = np.meshgrid(*beta_lin_set, indexing='ij')

    # Create linearized arrays for each concentration, stacked into one column per node:
    beta_test_set = np.asarray([bM.ravel() for bM in beta_M_SET]).T

    bif_space_M = [] # Matrix holding the parameter values and number of unique stable solutions
    sols_space_M = []

    if verbose:
        print(f'Solving for {beta_M_SET[0].ravel().shape} iterations...')

    for beta_set_i in beta_test_set:

        # Here we set di = 1.0, realizing the di value has no effect on the
        # steady-state since it can be divided through the rate equation when
        # solving for the root

        solsM, sol_M0_char, sol_0 = pnet.solve_probability_equms(constraint_inds=constraint_inds,
                                                                 constraint_vals=constraint_vals,
                                                                 signal_constr_vals=signal_constr_vals,
                                                                 d_base=1.0,
                                                                 n_base=n_base,
                                                                 beta_base=beta_set_i,
                                                                 N_space=N_search,
                                                                 search_tol=search_tol,
                                                                 sol_tol=sol_tol,
                                                                 N_round_sol=N_unique_sol_round,
                                                                 save_file=None,
                                                                 verbose=verbose,
                                                                 search_cycle_nodes_only=search_cycle_nodes_only
                                                                 )


        if len(solsM):
            num_sols = solsM.shape[1]
        else:
            num_sols = 0

        bif_space_M.append([*beta_set_i, num_sols])
        sols_space_M.append(solsM)

    return np.asarray(bif_space_M), sols_space_M