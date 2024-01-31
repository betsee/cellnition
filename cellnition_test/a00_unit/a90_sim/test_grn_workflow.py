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
    Test a gene regulatory network (GRN) workflow on a graph loaded
    from the network library.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.
    '''
    from cellnition.science.network_models.gene_networks import GeneNetworkModel
    from cellnition.science.network_enums import GraphType
    from cellnition.science.network_library import BasicTrinodeNet
    from cellnition.science.network_workflow import NetworkWorkflow

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    libg = BasicTrinodeNet()

    gmod = GeneNetworkModel(libg.N_nodes,
                            edges=libg.edges,
                            graph_type=GraphType.user
                            )  # This will be a straight-up GRN model

    fname_base = libg.name

    gmod.build_analytical_model(
        edge_types=libg.edge_types,
        add_interactions=libg.add_interactions,
        node_type_dict=libg.node_type_dict,
        pure_gene_edges_only=False
    )

    sim = NetworkWorkflow(save_path)

    graph_data = sim.work_frame(gmod,
                                save_path,
                                fname_base,
                                i_frame=0,
                                verbose=True,
                                reduce_dims=True,  # * True
                                Bi=2.0,  # * also use vector
                                ni=3.0,
                                di=1.0,
                                coi=0.0,
                                ki=10.0,
                                add_interactions=True,  # * False
                                edge_types=libg.edge_types,  # None
                                edge_type_search=False,  # * True
                                N_edge_search=3,
                                find_solutions=True,  # * False
                                knockout_experiments=True,  # *True
                                sol_search_tol=1.0e-15,
                                N_search_space=2,
                                N_round_sol=6,
                                N_round_unique_sol=1,
                                sol_unique_tol=1.0e-1,
                                sol_ko_tol=1.0e-1,
                                constraint_vals=[0.0, 0.0, 0.0],  # *signal vals
                                constraint_inds=gmod.signal_node_inds.copy(),  # * signal inds
                                update_string=None,  # * with string
                                pure_gene_edges_only=False,  # * True
                                node_type_dict=libg.node_type_dict,  # * None
                                solver_method='Root',  # * 'Powell'
                                extra_verbose=True
                                )

def test_grn_workflow_sfgraph(tmp_path) -> None:
    '''
    Test a random gene regulatory network (GRN) workflow.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Abstract path encapsulating a temporary directory unique to this unit
        test, created in the base temporary directory.
    '''
    from cellnition.science.network_workflow import NetworkWorkflow

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    sim = NetworkWorkflow(save_path)

    N_nodes = 5
    bi = 0.8
    gi = 0.15
    delta_in = 0.1
    delta_out = 0.0
    iframe = 0

    # randomly generate a scale-free graph:
    gmod, update_string, fname_base = sim.scalefree_graph_gen(N_nodes,
                                                              bi,
                                                              gi,
                                                              delta_in,
                                                              delta_out,
                                                              iframe)

    graph_data = sim.work_frame(gmod,
                                save_path,
                                fname_base,
                                i_frame=0,
                                verbose=True,
                                reduce_dims=False,  # * True
                                Bi=2.0,  # * also use vector
                                ni=3.0,
                                di=1.0,
                                coi=0.0,
                                ki=10.0,
                                add_interactions=True,  # * False
                                edge_types=None,  # None
                                edge_type_search=True,  # * True
                                N_edge_search=3,
                                find_solutions=True,  # * False
                                knockout_experiments=True,  # *True
                                sol_search_tol=1.0e-15,
                                N_search_space=2,
                                N_round_sol=6,
                                N_round_unique_sol=1,
                                sol_unique_tol=1.0e-1,
                                sol_ko_tol=1.0e-1,
                                constraint_vals = None, # *signal vals
                                constraint_inds = None, # * signal inds
                                update_string=update_string,  # * with string
                                pure_gene_edges_only=False,  # * True
                                node_type_dict=None,  # * None
                                solver_method='Root',  # * 'Powell'
                                extra_verbose=True
                                )

def test_grn_workflow_readfromfile(tmp_path) -> None:
    '''

    '''
    import os
    from cellnition.science.network_workflow import NetworkWorkflow
    from cellnition.science.network_enums import GraphType
    from cellnition.science.network_library import QuadStateNet
    from cellnition.science.network_models.gene_networks import GeneNetworkModel

    # Absolute or relative dirname of a test-specific temporary directory to
    # which "NetworkWorkflow" will emit GraphML and other files.
    save_path = str(tmp_path)

    # libg = QuadStateNet()
    # libg = FullQuadStateNet()
    libg = QuadStateNet()

    gmod = GeneNetworkModel(libg.N_nodes,
                            edges=libg.edges,
                            graph_type=GraphType.user
                            )  # This will be a straight-up GRN model

    fname_base = libg.name

    gmod.build_analytical_model(
        edge_types=libg.edge_types,
        add_interactions=libg.add_interactions,
        node_type_dict=libg.node_type_dict,
        pure_gene_edges_only=False
    )

    sim = NetworkWorkflow(save_path)

    graph_data = sim.work_frame(gmod,
                                save_path,
                                fname_base,
                                i_frame=0,
                                verbose=True,
                                reduce_dims=False,  # * True
                                Bi=2.0,  # * also use vector
                                ni=3.0,
                                di=1.0,
                                coi=0.0,
                                ki=10.0,
                                add_interactions=True,  # * False
                                edge_types=libg.edge_types,  # None
                                edge_type_search=False,  # * True
                                N_edge_search=3,
                                find_solutions=False,  # * False
                                knockout_experiments=False,  # *True
                                sol_search_tol=1.0e-15,
                                N_search_space=2,
                                N_round_sol=6,
                                N_round_unique_sol=1,
                                sol_unique_tol=1.0e-1,
                                sol_ko_tol=1.0e-1,
                                constraint_vals=[0.0, 0.0, 0.0],  # *signal vals
                                constraint_inds=gmod.signal_node_inds.copy(),  # * signal inds
                                update_string=None,  # * with string
                                pure_gene_edges_only=False,  # * True
                                node_type_dict=libg.node_type_dict,  # * None
                                solver_method='Root',  # * 'Powell'
                                extra_verbose=True
                                )



    filename = os.path.join(save_path, 'network_QuadStateNet.gml')

    gmod, updatestr, fnbase = sim.read_graph_from_file(filename,
                                                       add_interactions=True,
                                                       build_analytical=True,
                                                       i=0)

def test_network_library() -> None:
    '''
    Test the :mod:`cellnition.science.network_library` submodule.
    '''

    # Defer test-specific imports.
    from cellnition.science import network_library
    from cellnition.science.network_library import LibNet
    from cellnition.science.network_models.gene_networks import GeneNetworkModel
    from cellnition.science.network_enums import GraphType

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
        gmod = GeneNetworkModel(libn.N_nodes,
                            edges=libn.edges,
                            graph_type=GraphType.user
                            )