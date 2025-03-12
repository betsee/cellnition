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

    # Save model equations:
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
                                            verbose=True)

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

# def test_state_machine(tmp_path) -> None: