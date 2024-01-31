'''
This module has methods to search the state space or the parameter space of the model
for desired attributes.
'''
import numpy as np
from numpy import ndarray
from cellnition.science.network_models.probability_networks import ProbabilityNet

# FIXME: These need to be totally re-done with new probability networks...
#FIXME allow these to run with constraints on signals or nodes
# FIXME: we'd like to remove signal node edges and signal nodes from this search.

def multistability_search(pnet: ProbabilityNet,
                          N_multi: int = 1,
                          sol_tol: float = 1.0e-1,
                          N_iter: int = 100,
                          verbose: bool = True,
                          N_space: int = 2,
                          N_round_unique_sol: int = 1,
                          search_tol: float = 1.0e-15,
                          constraint_vals: list[float]|None = None,
                          constraint_inds: list[int]|None = None,
                          node_type_dict: dict|None = None,
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
        pnet.build_analytical_model(edge_types=edge_types,
                                    add_interactions=add_interactions,
                                    node_type_dict=node_type_dict)

        if constraint_vals is None or constraint_inds is None:
            sols_0 = pnet.optimized_phase_space_search(Ns=N_space,
                                                       cmax=1.5 * np.max(pnet.in_degree_sequence),
                                                       round_sol=N_round_sol,
                                                       tol=search_tol,
                                                       method="Root"
                                                       )

        else:
            sols_0 = pnet.constrained_phase_space_search(constraint_vals,
                                                         constraint_inds,
                                                         Ns=N_space,
                                                         cmax=1.5 * np.max(pnet.in_degree_sequence),
                                                         tol=search_tol,
                                                         round_sol=N_round_sol,
                                                         method='Root'
                                                         )

        solsM = pnet.find_attractor_sols(sols_0,
                                         tol=sol_tol,
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
                       constraint_inds: list[int] | None = None
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
    Blin = np.linspace(beta_min, beta_max, N_pts)

    param_lin_set = []

    for edj_i in range(pnet.regular_edges_index):
        param_lin_set.append(Blin*1) # append the linear K-vector choices for each edge

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
        print(f'Search cmax will be {cmax_multi * np.max(pnet.in_degree_sequence)}')

    for param_set_i in param_test_set:
        bvecti = param_set_i[0:pnet.N_edges].tolist()

        # Here we set di = 1.0, realizing the di value has no effect on the
        # steady-state since it can be divided through the rate equation when
        # solving for the root.
        pnet.create_parameter_vects(beta_base=bvecti, n_base=n_base, d_base=1.0, co=coi, ki=ki)

        if constraint_vals is None or constraint_inds is None:

            sols_0 = pnet.optimized_phase_space_search(Ns=N_search,
                                                       cmax=cmax_multi * np.max(pnet.in_degree_sequence),
                                                       round_sol=search_round_sol,
                                                       tol=1.0e-15,
                                                       method="Root"
                                                       )

        else:
            sols_0 = pnet.constrained_phase_space_search(constraint_vals,
                                                         constraint_inds,
                                                         Ns=N_search,
                                                         cmin=0.0,
                                                         cmax=cmax_multi * np.max(pnet.in_degree_sequence),
                                                         tol=1.0e-15,
                                                         round_sol=search_round_sol,
                                                         method='Root'
                                                         )

        solsM = pnet.find_attractor_sols(sols_0,
                                         tol=sol_tol,
                                         verbose=False,
                                         unique_sols=True,
                                         sol_round=N_unique_sol_round)

        if len(solsM):
            num_sols = solsM.shape[1]
        else:
            num_sols = 0

        bif_space_M.append([*pnet.beta_vect, num_sols])
        sols_space_M.append(solsM)

    return np.asarray(bif_space_M), sols_space_M