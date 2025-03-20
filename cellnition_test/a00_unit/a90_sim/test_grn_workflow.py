#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Simulation** unit tests.

This submodule unit tests the public API of the :mod:`cellnition.science`
subpackage.
'''

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To raise human-readable test errors, avoid importing from
# package-specific submodules at module scope.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ TESTS                              }....................

def test_continuous_net(tmp_path) -> None:
    '''
    Test the input space search and state transition network inference module
    with a library network.
    '''

    import os
    from cellnition.science.network_models.probability_networks import ProbabilityNet
    from cellnition.science.networks_toolbox.phase_space_viz import PhaseSpace
    from cellnition.science.network_models.network_library import TrinodeChain
    import matplotlib.pyplot as plt
    from cellnition.science.networks_toolbox.gene_knockout import GeneKnockout
    from cellnition.science.networks_toolbox.state_machine import StateMachine
    from cellnition.science.network_models.network_enums import (
                                                                 InterFuncType,
                                                                 CouplingType
                                                                 )

    # Study network:
    libg = TrinodeChain()

    # Study parameters:
    # Small value taken for zero:
    pmin = 1.0e-6

    # Number of linear points to search in state space:
    N_space = 2

    # Specify how the nodes should interact:
    multi_coupling_type = CouplingType.mix1

    # interaction_function_type = InterFuncType.hill
    interaction_function_type = InterFuncType.logistic

    return_saddles = True
    node_express_levels = 5.0

    # Set parameters for the network:
    dd = 1.0  # decay rate

    if interaction_function_type is InterFuncType.logistic:
        n_base = 50.0  # standard is 30.0, slope
        bb = 0.5  # standard is 0.5, centre

    else:
        n_base = 3.0  # standard is 3.0, slope
        bb = 2.0  # standard is 2.0, reciprocal centre

    net_name = libg.name
    save_path = os.path.join(tmp_path, f'{net_name}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Create the continuous differential-equation based model:
    pnet = ProbabilityNet(libg.N_nodes,
                          interaction_function_type=interaction_function_type,
                          node_expression_levels=node_express_levels)

    pnet.build_network_from_edges(libg.edges)
    pnet.characterize_graph()  # characterize the graph and set key params
    pnet.set_node_types()  # set the node types
    pnet.set_edge_types(libg.edge_types)  # set the edge types to the network

    # Get the adjacency matrices for this model:
    A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(libg.edge_types,
                                                                          pnet.edges_index,
                                                                          coupling_type=multi_coupling_type)
    pnet.build_analytical_model(A_add_s, A_mul_s)

    # Save the model equations:
    eqn_img = f'Eqn_{libg.name}.png'
    save_eqn_image = os.path.join(save_path, eqn_img)

    eqn_net_file = f'Eqns_{libg.name}.csv'
    save_eqn_csv = os.path.join(save_path, eqn_net_file)

    pnet.save_model_equations(save_eqn_image,
                              save_eqn_csv=save_eqn_csv,
                              substitute_node_labels=True)

    # Test out the gene knockout experiment:
    sigs = [0.0 for ii in pnet.input_node_inds]

    gko = GeneKnockout(pnet)
    knockout_sol_set, knockout_matrix, ko_header_o = gko.gene_knockout_ss_solve(
                                                                                Ns=2,
                                                                                tol=1.0e-15,
                                                                                d_base=dd,
                                                                                n_base=n_base,
                                                                                beta_base=bb,
                                                                                verbose=False,
                                                                                sol_tol=1.0e-1,
                                                                                save_file_basename=None,
                                                                                constraint_vals=None,
                                                                                constraint_inds=None,
                                                                                signal_constr_vals=sigs,
                                                                            )

    ko_file = f'knockoutArrays{libg.name}.png'
    save_ko = os.path.join(save_path, ko_file)
    fig, ax = gko.plot_knockout_arrays(knockout_sol_set, figsave=save_ko)
    plt.close(fig)

    # Create the Finite State Machine solver for this system:
    smach = StateMachine(pnet)

    solsM_all, charM_all, sols_list, states_dict, sig_test_set = smach.steady_state_solutions_search(beta_base=bb,
                                                                                                     n_base=n_base,
                                                                                                     d_base=dd,
                                                                                                     verbose=True,
                                                                                                     return_saddles=return_saddles,
                                                                                                     N_space=N_space,
                                                                                                     search_tol=1.0e-15,
                                                                                                     sol_tol=1.0e-3,
                                                                                                     search_main_nodes_only=False,
                                                                                                     order_by_distance=False
                                                                                                     )
    # test plotting the solution array:
    save_microarray_image = os.path.join(tmp_path, f'Microarray_{libg.name}.png')
    fig, ax = pnet.plot_sols_array(solsM_all,
                                   gene_inds=pnet.noninput_node_inds,
                                   figsave=save_microarray_image,
                                   cmap=None,
                                   save_format='png'
                                   )
    plt.close(fig)

    # Create the edges of the transition network:
    dt = 5.0e-3
    dt_samp = 0.1
    delta_sig = 60.0
    t_relax = 20.0  # originally 20
    match_tol = 1.0e-3
    remove_inaccessible_states = False

    save_graph_file = os.path.join(tmp_path, f'network_{libg.name}.gml')

    transition_edges_set, pert_edges_set, G_nx = smach.create_transition_network(states_dict,
                                                                                 sig_test_set,
                                                                                 solsM_all,
                                                                                 charM_all,
                                                                                 dt=dt,
                                                                                 delta_sig=delta_sig,
                                                                                 t_relax=t_relax,
                                                                                 dt_samp=dt_samp,
                                                                                 verbose=False,
                                                                                 match_tol=match_tol,
                                                                                 d_base=dd,
                                                                                 n_base=n_base,
                                                                                 beta_base=bb,
                                                                                 remove_inaccessible_states=remove_inaccessible_states,
                                                                                 save_graph_file=save_graph_file,

                                                                                 )

    nodes_list = list(G_nx.nodes())
    edges_list = list(G_nx.edges)

    save_perturbation_net_image = os.path.join(tmp_path, f'Pert_Net_{libg.name}.png')
    G_pert = smach.plot_state_perturbation_network(pert_edges_set,
                                                   charM_all,
                                                   nodes_listo=nodes_list,
                                                   save_file=save_perturbation_net_image,
                                                   graph_layout='dot',
                                                   mono_edge=False,
                                                   constraint=True,
                                                   concentrate=False,
                                                   rank='same'
                                                   )

    save_transition_net_image = os.path.join(tmp_path, f'Trans_Net_{libg.name}.png')
    G_gv = smach.plot_state_transition_network(nodes_list,
                                               edges_list,
                                               charM_all,
                                               save_file=save_transition_net_image,
                                               graph_layout='dot',
                                               mono_edge=False,
                                               constraint=True,
                                               concentrate=False,
                                               rank='same'
                                               )



    sigs = [0.0 for i in pnet.input_node_inds]

    ps = PhaseSpace(pnet)
    system_sols, dcdt_M_set, dcdt_dmag, c_lin_set, c_M_set = ps.brute_force_phase_space(N_pts=15,
                                                                                        constrained_inds=pnet.input_node_inds,
                                                                                        constrained_vals=sigs,
                                                                                        beta_base=bb,
                                                                                        n_base=n_base,
                                                                                        d_base=dd,
                                                                                        zer_thresh=0.01)
    fig, ax = plt.subplots()
    ax.quiver(c_M_set[0, 0], c_M_set[1, 0], dcdt_M_set[0, 0], dcdt_M_set[1, 0])
    plt.close(fig)

#
# def test_osmo_model() -> None:
#     '''
#
#     '''
#     import numpy as np
#     from cellnition.science.osmoadaptation.model_params import ModelParams
#     from cellnition.science.osmoadaptation.osmo_model import OsmoticCell
#
#     ocell = OsmoticCell()
#     p = ModelParams()  # Define a model params object
#
#     Np = 15
#     vol_vect = np.linspace(0.2 * p.vol_cell_o, 1.5 * p.vol_cell_o, Np)
#     # ni_vect = np.linspace(p.m_o_base*p.vol_cell_o, 1000.0*p.vol_cell_o, Np)
#     ni_vect = np.linspace(0.25 * p.m_o_base * p.vol_cell_o, 1500.0 * p.vol_cell_o, Np)
#     mo_vect = np.linspace(p.m_o_base, 1000.0, Np)
#
#     # VV, NN, MM = np.meshgrid(vol_vect, ni_vect, mo_vect, indexing='ij')
#     MM, NN, VV = np.meshgrid(mo_vect, ni_vect, vol_vect, indexing='ij')
#
#     dVdt_vect, dndt_vect, _ = ocell.state_space_gen(MM.ravel(),
#                                                     VV.ravel(),
#                                                     NN.ravel(),
#                                                     p.m_i_gly,
#                                                     p.d_wall,
#                                                     p.Y_wall,
#                                                     p,
#                                                     synth_gly=True
#                                                     )
#
#     # Compute steady-state solutions:
#     # Need to calculate solutions over the full domain first, then find solutinos that match the region criteria:
#     Vss_vect = ocell.osmo_vol_steady_state(MM.ravel(), NN.ravel(), p.Y_wall, p.d_wall, p)
#
#
# def test_grn_workflow_sfgraph(tmp_path) -> None:
#     '''
#     Test generation of a randomly generated scale-free gene regulatory network model.
#
#     Parameters
#     ----------
#     tmp_path : pathlib.Path
#         Abstract path encapsulating a temporary directory unique to this unit
#         test, created in the base temporary directory.
#     '''
#     from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
#     from cellnition.science.network_workflow import NetworkWorkflow
#
#     # Absolute or relative dirname of a test-specific temporary directory to
#     # which "NetworkWorkflow" will emit GraphML and other files.
#     save_path = str(tmp_path)
#
#     netflow = NetworkWorkflow(save_path)
#
#     N_nodes = 5
#     bi = 0.8
#     gi = 0.15
#     delta_in = 0.1
#     delta_out = 0.0
#     iframe = 0
#
#     # randomly generate a scale-free graph:
#     pnet, update_string, fname_base = netflow.scalefree_graph_gen(N_nodes,
#                                                                   bi,
#                                                                   gi,
#                                                                   delta_in,
#                                                                   delta_out,
#                                                                   iframe,
#                                                                   interaction_function_type=InterFuncType.logistic,
#                                                                   coupling_type=CouplingType.mixed)
#
# def test_grn_workflow_bingraph(tmp_path) -> None:
#     '''
#     Test generation of a randomly generated binomial gene regulatory network model.
#
#     Parameters
#     ----------
#     tmp_path : pathlib.Path
#         Abstract path encapsulating a temporary directory unique to this unit
#         test, created in the base temporary directory.
#     '''
#     from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
#     from cellnition.science.network_workflow import NetworkWorkflow
#
#     # Absolute or relative dirname of a test-specific temporary directory to
#     # which "NetworkWorkflow" will emit GraphML and other files.
#     save_path = str(tmp_path)
#
#     netflow = NetworkWorkflow(save_path)
#
#     N_nodes = 5
#     p_edge = 0.5
#     iframe = 0
#
#     # randomly generate a scale-free graph:
#     pnet, update_string, fname_base = netflow.binomial_graph_gen(N_nodes,
#                                                                   p_edge,
#                                                                   iframe,
#                                                                   interaction_function_type=InterFuncType.logistic,
#                                                                   coupling_type=CouplingType.mixed)
#
# def test_grn_workflow_readwritefromfile(tmp_path) -> None:
#     '''
#     Test writing and reading a network model to file.
#
#     Parameters
#     ----------
#     tmp_path : pathlib.Path
#         Abstract path encapsulating a temporary directory unique to this unit
#         test, created in the base temporary directory.
#
#     '''
#     import os
#     from cellnition.science.network_workflow import NetworkWorkflow
#     from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
#
#     # Absolute or relative dirname of a test-specific temporary directory to
#     # which "NetworkWorkflow" will emit GraphML and other files.
#     save_path = str(tmp_path)
#
#     netflow = NetworkWorkflow(save_path)
#
#     N_nodes = 5
#     bi = 0.8
#     gi = 0.15
#     delta_in = 0.1
#     delta_out = 0.0
#     iframe = 0
#
#     interfunctype = InterFuncType.logistic
#     couplingtype = CouplingType.mixed
#
#     if interfunctype is InterFuncType.logistic:
#         d_base = 1.0
#         n_base = 15.0
#         beta_base = 0.25
#     else:
#         d_base = 1.0
#         n_base = 3.0
#         beta_base = 5.0
#
#     # randomly generate a scale-free graph:
#     pnet, update_string, fname_base = netflow.scalefree_graph_gen(N_nodes,
#                                                                   bi,
#                                                                   gi,
#                                                                   delta_in,
#                                                                   delta_out,
#                                                                   iframe,
#                                                                   interaction_function_type=interfunctype,
#                                                                   coupling_type=couplingtype)
#
#     # get random edge types:
#     edge_types = pnet.get_edge_types()
#
#
#     graph_dat = netflow.work_frame(pnet,
#                                    save_path,
#                                    fname_base,
#                                    i_frame=0,
#                                    verbose=False,
#                                    reduce_dims=False,
#                                    beta_base=beta_base,
#                                    n_base=n_base,
#                                    d_base=d_base,
#                                    edge_types=edge_types,
#                                    edge_type_search=False,
#                                    edge_type_search_iterations=3,
#                                    find_solutions=False,
#                                    knockout_experiments=False,
#                                    sol_search_tol=1.0e-15,
#                                    N_search_space=3,
#                                    sol_unique_tol=1.0e-1,
#                                    sol_ko_tol=1.0e-1,
#                                    constraint_vals=None,
#                                    constraint_inds=None,
#                                    signal_constr_vals=None,
#                                    update_string=update_string,
#                                    node_type_dict=None,
#                                    extra_verbose=False,
#                                    coupling_type=couplingtype
#                                    )
#
#
#
#     filename = os.path.join(save_path, f'network_{fname_base}.gml')
#
#     gmod, updatestr, fnbase = netflow.read_graph_from_file(filename, interaction_function_type=interfunctype,
#                              coupling_type=couplingtype, i=0)
#

#
# def test_time_sim(tmp_path) -> None:
#     '''
#     Test the time simulation capabilities of the probability network.
#
#     Parameters
#     ----------
#     tmp_path : pathlib.Path
#         Abstract path encapsulating a temporary directory unique to this unit
#         test, created in the base temporary directory.
#
#     '''
#     import numpy as np
#     from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
#     from cellnition.science.network_models.network_library import TrinodeNet
#     from cellnition.science.network_workflow import NetworkWorkflow
#
#     libg = TrinodeNet()
#
#     # Absolute or relative dirname of a test-specific temporary directory to
#     # which "NetworkWorkflow" will emit GraphML and other files.
#     save_path = str(tmp_path)
#
#     netflow = NetworkWorkflow(save_path)
#
#     interfunctype = InterFuncType.hill
#
#     if interfunctype is InterFuncType.logistic:
#         d_base = 1.0
#         n_base = 15.0
#         beta_base = 0.25
#     else:
#         d_base = 1.0
#         n_base = 3.0
#         beta_base = 5.0
#
#     pnet, update_string, fname_base = netflow.make_network_from_edges(libg.edges,
#                                                                       edge_types=libg.edge_types,
#                                                                       interaction_function_type=interfunctype,
#                                                                       coupling_type=CouplingType.specified,
#                                                                       network_name=libg.name,
#                                                                       i=0)
#
#     dt = 1.0e-3
#     dt_samp = 0.15
#
#     sig_inds = pnet.input_node_inds
#     N_sigs = len(sig_inds)
#
#     space_sig = 25.0  # spacing between two signal perturbations
#     delta_sig = 10.0  # Time for a signal perturbation
#
#     sig_times = [(space_sig + space_sig * i + delta_sig * i, delta_sig + space_sig + space_sig * i + delta_sig * i) for
#                  i in range(N_sigs)]
#
#     tend = sig_times[-1][1] + space_sig
#
#     sig_base_vals = [0.0, 0.0, 0.0]
#     sig_mags = [(int(sigi) + pnet.p_min, int(not (int(sigi))) + pnet.p_min) for sigi in sig_base_vals]
#
#     cvecti = np.zeros(pnet.N_nodes) + pnet.p_min
#
#     # Get the full time vector and the sampled time vector (tvectr)
#     tvect, tvectr = pnet.make_time_vects(tend, dt, dt_samp)
#
#     c_signals = pnet.make_pulsed_signals_matrix(tvect, sig_inds, sig_times, sig_mags)
#
#     ctime = pnet.run_time_sim(tvect,
#                               tvectr,
#                               cvecti,
#                               sig_inds=sig_inds,
#                               sig_vals=c_signals,
#                               constrained_inds=None,
#                               constrained_vals=None,
#                               d_base=d_base,
#                               n_base=n_base,
#                               beta_base=beta_base
#                              )
