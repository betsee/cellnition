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
    if all graphs can be loaded, characterized, and used to build a
    Boolean network base class.
    '''

    # Defer test-specific imports.
    from cellnition.science.network_models import network_library
    from cellnition.science.network_models.network_library import LibNet
    from cellnition.science.network_models.network_enums import InterFuncType
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

        # Let's ensure we can build a Boolean model from this
        # imported graph:
        bn = BooleanNet()  # instantiate bool net solver
        bn.build_network_from_edges(libn.edges)  # build basic graph
        bn.characterize_graph()  # characterize the graph and set key params
        bn.set_node_types()

        bn.set_edge_types(libn.edge_types)  # set the edge types to the network

def test_basic_network(tmp_path) -> None:
    '''
    Test the BasicNetwork module to ensure random and defined graphs can
    be generated, characterized, and plotted.
    '''
    import os
    import matplotlib.pyplot as plt
    from cellnition.science.network_models.basic_network import BasicNet
    from cellnition.science.network_models.network_enums import GraphType
    from cellnition.science.networks_toolbox.netplot import plot_network

    graph_types = [GraphType.scale_free, GraphType.random]
    N_nodes = 10  # make a network with 10 nodes

    for gt in graph_types:

        bn = BasicNet()

        # Generate a randomly-created graph with scale free or random architecture:
        bn.randomly_generate_special_network(N_nodes,
                                             b_param=0.15,
                                             g_param=0.8,
                                             delta_in=0.0,
                                             delta_out=0.0,
                                             p_edge=0.5,
                                             graph_type=gt
                                             )
        bn.characterize_graph()  # characterize the graph and set key params
        bn.set_node_types() # set default node types
        # obtain random activator or inhibitor edge types:
        edge_types = bn.get_edge_types(p_acti=0.5, set_selfloops_acti=True)
        bn.set_edge_types(edge_types) # set the edge types

        # Save a plot of the graph:
        graph_net_a = f'hier_graph_test_a.png'
        save_graph_net_hier = os.path.join(tmp_path, graph_net_a)

        # plot the network:
        gp = plot_network(bn.nodes_list,
                          bn.edges_list,
                          bn.node_types,
                          bn.edge_types,
                          node_vals=bn.hier_node_level,
                          val_cmap='viridis_r',
                          save_path=save_graph_net_hier,
                          layout='dot',
                          rev_font_color=False,
                          label_edges=False,
                          net_font_name='DejaVu Sans Bold',
                          node_font_size=24,
                          edge_width=2.0,
                          nde_outline='Black',
                          arrowsize=2.0
                          )

        # Plot the graph's degree distributions:
        fig, ax = bn.plot_degree_distributions()
        plt.close(fig)

        # Simpler type of graph plot:
        graph_net_b = f'hier_graph_test_b.png'
        save_graph_net = os.path.join(tmp_path, graph_net_b)
        bn.save_network_image(save_graph_net, use_dot_layout=False)
