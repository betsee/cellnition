#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Boolean** GRN network unit tests.

This submodule unit tests the public API of the
:mod:`cellnition.science.network_models.boolean_networks`
subpackage.
'''

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To raise human-readable test errors, avoid importing from
# package-specific submodules at module scope.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ TESTS                              }....................
def test_boolean_net(tmp_path) -> None:
    '''
    Builds an analytic and numerical Boolean network model that can be used to model
    a regulatory network's output.
    '''
    import os
    from cellnition.science.network_models.network_library import TrinodeChain
    from cellnition.science.network_models.network_enums import CouplingType
    from cellnition.science.network_models.boolean_networks import BooleanNet
    from cellnition_test._util.pytci import is_ci_github_actions
    from cellnition.science.networks_toolbox.boolean_state_machine import BoolStateMachine

    multi_coupling_type = CouplingType.mix1  # activators combine as "OR" and inhibitors "AND"
    constitutive_express = False  # activators present "AND" inhibitors absent for expression, when "False"

    libg = TrinodeChain()

    bn = BooleanNet()  # instantiate bool net solver
    bn.build_network_from_edges(libg.edges)  # build basic graph from library import
    bn.characterize_graph()  # characterize the graph and set key params
    bn.set_node_types() # set the node types
    bn.set_edge_types(libg.edge_types)  # set the edge types to the network

    # Build the Boolean Network model
    c_vect_s, A_bool_s, A_bool_f = bn.build_boolean_model(use_node_name=True,
                                                          multi_coupling_type=multi_coupling_type,
                                                          constitutive_express=constitutive_express)

    # If this test is *NOT* currently being run under a remote GitHub Actions-
    # based continuous integration (CI) workflow, this test is running locally.
    # In this case, save model equations. Doing so requires LaTeX and thus
    # TeXLive -- a third-party dependency that is typically several gigabytes
    # (GB) in size and thus infeasible to install in CI.
    if not is_ci_github_actions():
        # tmp_path = '/home/pietakio/Dropbox/Levin_2024/Tests'
        save_eqns_img = os.path.join(tmp_path, f'eqns_{libg.name}')
        bn.save_model_equations(save_eqns_img)

    # Create a state transition diagram for a single signal value of all zeros:
    sigs = [0 for i in bn.input_node_inds] # initial signal vals vector
    cc_o = [0 for i in bn.nodes_index] # initial concentration vector

    boolGG, boolpos = bn.bool_state_space(
                                            A_bool_f,
                                            constraint_inds=None,
                                            constraint_vals=None,
                                            signal_constr_vals=sigs,
                                            search_main_nodes_only=True,
                                            n_max_steps=2*len(bn.main_nodes),
                                            node_num_max=bn.N_nodes,
                                            )

    # Compute a pseudo-time sequence:
    solsv, cc_i, sol_char, motif = bn.net_sequence_compute(cc_o,
                                                           A_bool_f,
                                                           n_max_steps=len(bn.main_nodes) * 2,
                                                           constraint_inds=bn.input_node_inds,
                                                           constraint_vals=sigs,
                                                           verbose=True,
                                                           )

    # Solve and characterize steady-state solutions at an input signal:
    sol_M, sol_char = bn.solve_system_equms(A_bool_f,
                                            constraint_inds=None,
                                            constraint_vals=None,
                                            signal_constr_vals=sigs,
                                            search_main_nodes_only=False,
                                            n_max_steps=2 * len(bn.main_nodes),
                                            node_num_max=bn.N_nodes,
                                            verbose=False
                                            )

    # Next we build a Finite State Machine builder for the Boolean system:
    bsm = BoolStateMachine(bn)

    # Solve for the full equilibrium state matrix:
    n_max_steps = len(bn.main_nodes) * 2

    (solsM_all,
     charM_all,
     sols_list,
     states_dict,
     sig_test_set) = bsm.steady_state_solutions_search(verbose=False,
                                                       search_main_nodes_only=False,
                                                       n_max_steps=n_max_steps,
                                                       order_by_distance=False,
                                                       node_num_max=bn.N_nodes,
                                                       output_nodes_only=False
                                                       )

    # Plot the sols array
    bool_sols_a = f'bool_solM_{libg.name}.png'
    save_bool_sols = os.path.join(tmp_path, bool_sols_a)

    bn.plot_sols_array(solsM_all,
                       gene_inds=bn.noninput_node_inds,
                       figsave=save_bool_sols,
                       cmap=None,
                       save_format='png',
                       figsize=(20, 20))

    save_inputs_image = os.path.join(tmp_path, f'Bool_Inputs_{libg.name}_smach.png')
    y_input_labels = [bn.nodes_list[ni] for ni in bn.input_node_inds]
    x_input_labels = [f'I{ni}' for ni, _ in enumerate(sig_test_set)]

    fig, ax = bn.plot_pixel_matrix(sig_test_set.T,
                                   x_input_labels,
                                   y_input_labels,
                                   figsave=save_inputs_image,
                                   cmap=None,
                                   figsize=(10, 10),
                                   fontsize=24
                                   )

    # Create the network finite state machines:
    gNFSM_edges_set, eNFSM_edges_set, GG = bsm.create_transition_network(
                                                            states_dict,
                                                            sig_test_set,
                                                            solsM_all,
                                                            verbose=False,
                                                            remove_inaccessible_states=False,
                                                            save_graph_file=None,
                                                            n_max_steps=n_max_steps,
                                                            output_nodes_only=False
                                                        )

    # Save images of the NFSMs:
    nodes_list = list(GG.nodes())
    edges_list = list(GG.edges)

    save_perturbation_net_image = os.path.join(tmp_path, f'Bool_Pert_Net_{libg.name}.png')
    G_pert = bsm.plot_state_perturbation_network(eNFSM_edges_set,
                                                 charM_all,
                                                 nodes_listo=nodes_list,
                                                 save_file=save_perturbation_net_image,
                                                 graph_layout='dot',
                                                 mono_edge=True,
                                                 constraint=True,
                                                 concentrate=False,
                                                 node_colors=None,
                                                 cmap_str='RdBu',
                                                 transp_str='60',
                                                 rank='same'
                                                 )

    save_transition_net_image = os.path.join(tmp_path, f'Bool_Trans_Net_{libg.name}.png')
    G_gv = bsm.plot_state_transition_network(nodes_list,
                                             edges_list,
                                             charM_all,
                                             save_file=save_transition_net_image,
                                             graph_layout='dot',
                                             mono_edge=True,
                                             constraint=True,
                                             concentrate=False,
                                             node_colors=None,
                                             rank='same',
                                             cmap_str='RdBu',
                                             transp_str='60'

                                             )

# def test_state_machine(tmp_path) -> None: