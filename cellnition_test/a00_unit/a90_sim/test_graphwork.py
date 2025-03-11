#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Graph/Network** unit tests.

This submodule unit tests the functionality for loading, generating, and analyzing
graphs/networks, which are classes of the public API of the
:mod:`cellnition.science.network_models` and
:mod:`cellnition.science.networks_toolbox`subpackages.
'''

# ....................{ IMPORTS                            }....................
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# WARNING: To raise human-readable test errors, avoid importing from
# package-specific submodules at module scope.
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ....................{ TESTS                              }....................
def test_network_library(tmp_path) -> None:
    '''
    Test the :mod:`cellnition.science.network_library` submodule to determine
    if all graphs can be loaded, characterized, and used to build both
    fully-continuous and Boolean network models.
    '''

    # Defer test-specific imports.
    from cellnition.science.network_models import network_library
    from cellnition.science.network_models.network_library import LibNet
    from cellnition.science.network_workflow import NetworkWorkflow
    from cellnition.science.network_models.network_enums import CouplingType, InterFuncType
    from cellnition.science.network_models.boolean_networks import BooleanNet

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

        # Determine if the basic features of the library graph can be loaded;
        # we don't make an analytical model as it'll take too much time:
        pnet, update_string, fname_base = netflow.make_network_from_edges(libn.edges,
                                                                          edge_types=libn.edge_types,
                                                                          interaction_function_type=interfunctype,
                                                                          coupling_type=CouplingType.mix1,
                                                                          network_name=libn.name,
                                                                          build_analytical_model=False,
                                                                          i=0)

        # Let's also ensure we can build a full Boolean model from this
        # imported graph:
        bn = BooleanNet()  # instantiate bool net solver
        bn.build_network_from_edges(libn.edges)  # build basic graph
        bn.characterize_graph()  # characterize the graph and set key params
        bn.set_node_types()

        bn.set_edge_types(libn.edge_types)  # set the edge types to the network