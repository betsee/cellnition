#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2021-2022 Ionovate.
# See "LICENSE" for further details.

'''
**Simulation** unit tests.

This submodule unit tests the public API of the :mod:`cellnition.science`
submodule.
'''

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To raise human-readable test errors, avoid importing from
# package-specific submodules at module scope.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ TESTS                              }....................

def test_state_machine(tmp_path) -> None:
    '''
    Test the input space search and
    state transition network inference module with a library network.
    '''
    import os
    from cellnition.science.network_models.network_library import BinodeNet
    from cellnition.science.network_workflow import NetworkWorkflow
    from cellnition.science.networks_toolbox.state_machine import StateMachine
    from cellnition.science.network_models.network_enums import InterFuncType, CouplingType

    libg = BinodeNet() # generate the library-derived network

    save_path = str(tmp_path) # temporary directory to write to
    netflow = NetworkWorkflow(save_path) # instantiate network work flow object

    interfunctype = InterFuncType.logistic

    pnet, update_string, fname_base = netflow.make_network_from_edges(libg.edges,
                                                                      edge_types=libg.edge_types,
                                                                      interaction_function_type=interfunctype,
                                                                      coupling_type=CouplingType.mixed,
                                                                      network_name=libg.name,
                                                                      i=0)

    if interfunctype is InterFuncType.logistic:
        d_base = 1.0
        n_base = 15.0
        beta_base = 0.25
    else:
        d_base = 1.0
        n_base = 3.0
        beta_base = 5.0

    save_graph_file = os.path.join(save_path, f'graph_{fname_base}.gml')
    save_file = os.path.join(save_path, f'transnet_{fname_base}.png')
    save_file_pert = os.path.join(save_path, f'pertnet_{fname_base}.png')

    smach = StateMachine(pnet) # Instantiate a state machine

    G = smach.run_state_machine(beta_base=beta_base,
                                n_base=n_base,
                                d_base=d_base,
                                verbose=False,
                                return_saddles=True,
                                N_space=2,
                                search_tol=1.0e-15,
                                sol_tol=1.0e-2,
                                N_round_sol=1,
                                dt=5.0e-3,
                                tend=80.0,
                                space_sig=25.0,
                                delta_sig=25.0,
                                t_relax=10.0,
                                dt_samp=0.15,
                                match_tol=0.05,
                                save_graph_file=save_graph_file,
                                save_transition_net_image=save_file,
                                save_perturbation_net_image=save_file_pert,
                                graph_layout='dot'
                                )

    dist_M = smach.get_state_distance_matrix(smach.solsM_all)

    input_list = ['I0', 'I1', 'I2', 'I0', 'I3']
    starting_state = 0
    tvectr, c_time, matched_states, phase_inds = smach.sim_time_trajectory(starting_state,
                                                              smach.solsM_all,
                                                              input_list,
                                                              smach.sig_test_set,
                                                              dt=1.0e-3,
                                                              dt_samp=0.1,
                                                              input_hold_duration=30.0,
                                                              t_wait=10.0,
                                                              verbose=True,
                                                              match_tol=0.05,
                                                              d_base=d_base,
                                                              n_base=n_base,
                                                              beta_base=beta_base,
                                                              )

    savefig = os.path.join(save_path, 'time_traj.png')
    fig, ax = smach.plot_time_trajectory(c_time, tvectr, phase_inds,
                                         matched_states,
                                         smach.charM_all,
                                         figsize=(10, 4),
                                         state_label_offset=0.01,
                                         glyph_zoom=0.1,
                                         glyph_alignment=(-0.3, -0.4),
                                         fontsize='medium',
                                         save_file=savefig)

def test_osmo_model() -> None:
    '''

    '''
    import numpy as np
    from cellnition.science.osmoadaptation.model_params import ModelParams
    from cellnition.science.osmoadaptation.osmo_model import OsmoticCell

    ocell = OsmoticCell()
    p = ModelParams()  # Define a model params object

    Np = 15
    vol_vect = np.linspace(0.2 * p.vol_cell_o, 1.5 * p.vol_cell_o, Np)
    # ni_vect = np.linspace(p.m_o_base*p.vol_cell_o, 1000.0*p.vol_cell_o, Np)
    ni_vect = np.linspace(0.25 * p.m_o_base * p.vol_cell_o, 1500.0 * p.vol_cell_o, Np)
    mo_vect = np.linspace(p.m_o_base, 1000.0, Np)

    # VV, NN, MM = np.meshgrid(vol_vect, ni_vect, mo_vect, indexing='ij')
    MM, NN, VV = np.meshgrid(mo_vect, ni_vect, vol_vect, indexing='ij')

    dVdt_vect, dndt_vect, _ = ocell.state_space_gen(MM.ravel(),
                                                    VV.ravel(),
                                                    NN.ravel(),
                                                    p.m_i_gly,
                                                    p.d_wall,
                                                    p.Y_wall,
                                                    p,
                                                    synth_gly=True
                                                    )

    # Compute steady-state solutions:
    # Need to calculate solutions over the full domain first, then find solutinos that match the region criteria:
    Vss_vect = ocell.osmo_vol_steady_state(MM.ravel(), NN.ravel(), p.Y_wall, p.d_wall, p)

def test_grn_workflow_libgraph(tmp_path) -> None:
    '''
    Test a network (GRN) workflow on a graph loaded
    from the network library.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.
    '''
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
    from cellnition.science.network_models.network_library import TrinodeNet
    from cellnition.science.network_workflow import NetworkWorkflow

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    libg = TrinodeNet()

    netflow = NetworkWorkflow(save_path)

    interfunctype = InterFuncType.logistic

    if interfunctype is InterFuncType.logistic:
        d_base = 1.0
        n_base = 15.0
        beta_base = 0.25
    else:
        d_base = 1.0
        n_base = 3.0
        beta_base = 5.0

    pnet, update_string, fname_base = netflow.make_network_from_edges(libg.edges,
                                                                      edge_types=libg.edge_types,
                                                                      interaction_function_type=interfunctype,
                                                                      coupling_type=CouplingType.mixed,
                                                                      network_name=libg.name,
                                                                      i=0)

    graph_dat = netflow.work_frame(pnet,
                                   save_path,
                                   fname_base,
                                   i_frame=0,
                                   verbose=False,
                                   reduce_dims=False,
                                   beta_base=beta_base,
                                   n_base=n_base,
                                   d_base=d_base,
                                   edge_types=libg.edge_types,
                                   edge_type_search=True,
                                   edge_type_search_iterations=3,
                                   find_solutions=True,
                                   knockout_experiments=True,
                                   sol_search_tol=1.0e-15,
                                   N_search_space=3,
                                   N_round_unique_sol=1,
                                   sol_unique_tol=1.0e-1,
                                   sol_ko_tol=1.0e-1,
                                   constraint_vals=None,
                                   constraint_inds=None,
                                   signal_constr_vals=None,
                                   update_string=update_string,
                                   node_type_dict=None,
                                   extra_verbose=False,
                                   coupling_type=CouplingType.mixed
                                   )

def test_grn_workflow_sfgraph(tmp_path) -> None:
    '''
    Test generation of a randomly generated scale-free gene regulatory network model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.
    '''
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
    from cellnition.science.network_workflow import NetworkWorkflow

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    netflow = NetworkWorkflow(save_path)

    N_nodes = 5
    bi = 0.8
    gi = 0.15
    delta_in = 0.1
    delta_out = 0.0
    iframe = 0

    # randomly generate a scale-free graph:
    pnet, update_string, fname_base = netflow.scalefree_graph_gen(N_nodes,
                                                                  bi,
                                                                  gi,
                                                                  delta_in,
                                                                  delta_out,
                                                                  iframe,
                                                                  interaction_function_type=InterFuncType.logistic,
                                                                  coupling_type=CouplingType.mixed)

def test_grn_workflow_bingraph(tmp_path) -> None:
    '''
    Test generation of a randomly generated binomial gene regulatory network model.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.
    '''
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
    from cellnition.science.network_workflow import NetworkWorkflow

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    netflow = NetworkWorkflow(save_path)

    N_nodes = 5
    p_edge = 0.2
    iframe = 0

    # randomly generate a scale-free graph:
    pnet, update_string, fname_base = netflow.binomial_graph_gen(N_nodes,
                                                                  p_edge,
                                                                  iframe,
                                                                  interaction_function_type=InterFuncType.logistic,
                                                                  coupling_type=CouplingType.mixed)

def test_grn_workflow_readwritefromfile(tmp_path) -> None:
    '''
    Test writing and reading a network model to file.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.

    '''
    import os
    from cellnition.science.network_workflow import NetworkWorkflow
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    netflow = NetworkWorkflow(save_path)

    N_nodes = 5
    bi = 0.8
    gi = 0.15
    delta_in = 0.1
    delta_out = 0.0
    iframe = 0

    interfunctype = InterFuncType.logistic
    couplingtype = CouplingType.mixed

    if interfunctype is InterFuncType.logistic:
        d_base = 1.0
        n_base = 15.0
        beta_base = 0.25
    else:
        d_base = 1.0
        n_base = 3.0
        beta_base = 5.0

    # randomly generate a scale-free graph:
    pnet, update_string, fname_base = netflow.scalefree_graph_gen(N_nodes,
                                                                  bi,
                                                                  gi,
                                                                  delta_in,
                                                                  delta_out,
                                                                  iframe,
                                                                  interaction_function_type=interfunctype,
                                                                  coupling_type=couplingtype)

    # get random edge types:
    edge_types = pnet.get_edge_types()


    graph_dat = netflow.work_frame(pnet,
                                   save_path,
                                   fname_base,
                                   i_frame=0,
                                   verbose=False,
                                   reduce_dims=False,
                                   beta_base=beta_base,
                                   n_base=n_base,
                                   d_base=d_base,
                                   edge_types=edge_types,
                                   edge_type_search=False,
                                   edge_type_search_iterations=3,
                                   find_solutions=False,
                                   knockout_experiments=False,
                                   sol_search_tol=1.0e-15,
                                   N_search_space=3,
                                   N_round_unique_sol=1,
                                   sol_unique_tol=1.0e-1,
                                   sol_ko_tol=1.0e-1,
                                   constraint_vals=None,
                                   constraint_inds=None,
                                   signal_constr_vals=None,
                                   update_string=update_string,
                                   node_type_dict=None,
                                   extra_verbose=False,
                                   coupling_type=couplingtype
                                   )



    filename = os.path.join(save_path, f'network_{fname_base}.gml')

    gmod, updatestr, fnbase = netflow.read_graph_from_file(filename, interaction_function_type=interfunctype,
                             coupling_type=couplingtype, i=0)

def test_network_library(tmp_path) -> None:
    '''
    Test the :mod:`cellnition.science.network_library` submodule.
    '''

    # Defer test-specific imports.
    from cellnition.science.network_models import network_library
    from cellnition.science.network_models.network_library import LibNet
    from cellnition.science.network_workflow import NetworkWorkflow
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType

    # Tuple of all "LibNet" subclasses, defined as the tuple comprehension of...
    LIB_NETS: tuple[type[LibNet]] = tuple(
        attr_value
        # For the value of each attribute defined by this submodule...
        for attr_value in network_library.__dict__.values()
        # If this attribute that is a "LibNet" subclass.
        if (
            isinstance(attr_value, type) and
            issubclass(attr_value, LibNet) and
            attr_value is not LibNet
        )
    )

    for lib_net in LIB_NETS:
        libn = lib_net()
        interfunctype = InterFuncType.logistic

        save_path = str(tmp_path)

        netflow = NetworkWorkflow(save_path)

        pnet, update_string, fname_base = netflow.make_network_from_edges(libn.edges,
                                                                          edge_types=libn.edge_types,
                                                                          interaction_function_type=interfunctype,
                                                                          coupling_type=CouplingType.mixed,
                                                                          network_name=libn.name,
                                                                          i=0)


def test_time_sim(tmp_path) -> None:
    '''
    Test the time simulation capabilities of the probability network.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.

    '''
    import numpy as np
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
    from cellnition.science.network_models.network_library import TrinodeNet
    from cellnition.science.network_workflow import NetworkWorkflow

    libg = TrinodeNet()

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    netflow = NetworkWorkflow(save_path)

    interfunctype = InterFuncType.hill

    if interfunctype is InterFuncType.logistic:
        d_base = 1.0
        n_base = 15.0
        beta_base = 0.25
    else:
        d_base = 1.0
        n_base = 3.0
        beta_base = 5.0

    pnet, update_string, fname_base = netflow.make_network_from_edges(libg.edges,
                                                                      edge_types=libg.edge_types,
                                                                      interaction_function_type=interfunctype,
                                                                      coupling_type=CouplingType.specified,
                                                                      network_name=libg.name,
                                                                      i=0)

    dt = 1.0e-3
    dt_samp = 0.15

    sig_inds = pnet.input_node_inds
    N_sigs = len(sig_inds)

    space_sig = 25.0  # spacing between two signal perturbations
    delta_sig = 10.0  # Time for a signal perturbation

    sig_times = [(space_sig + space_sig * i + delta_sig * i, delta_sig + space_sig + space_sig * i + delta_sig * i) for
                 i in range(N_sigs)]

    tend = sig_times[-1][1] + space_sig

    sig_base_vals = [0.0, 0.0, 0.0]
    sig_mags = [(int(sigi) + pnet.p_min, int(not (int(sigi))) + pnet.p_min) for sigi in sig_base_vals]

    cvecti = np.zeros(pnet.N_nodes) + pnet.p_min

    # Get the full time vector and the sampled time vector (tvectr)
    tvect, tvectr = pnet.make_time_vects(tend, dt, dt_samp)

    c_signals = pnet.make_pulsed_signals_matrix(tvect, sig_inds, sig_times, sig_mags)

    ctime = pnet.run_time_sim(tvect,
                              tvectr,
                              cvecti,
                              sig_inds=sig_inds,
                              sig_vals=c_signals,
                              constrained_inds=None,
                              constrained_vals=None,
                              d_base=d_base,
                              n_base=n_base,
                              beta_base=beta_base
                             )