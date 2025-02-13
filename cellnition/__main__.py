#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2024 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
Non-reusable **Python app entry-point wrapper** (i.e., submodule
unconditionally running this app as an intentional side effect of importation
by the active Python interpreter itself and thus *not* intended to be imported
from anywhere within this codebase).

This submodule is implicitly run by the active Python interpreter when the
fully-qualified name of the top-level package directly containing this
submodule is passed as the value of the ``--m`` option to this interpreter on
the command line (e.g., ``python3 -m cellnition``).

'''


# ....................{ IMPORTS                            }....................
import sys
import os
import csv
import copy
import itertools
from collections import OrderedDict
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib import colormaps
import matplotlib.image as image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata
import networkx as nx

from cellnition.science.network_models.basic_network import BasicNet
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             InterFuncType,
                                                             CouplingType,
                                                             EquilibriumType
                                                            )
from cellnition.science.network_workflow import NetworkWorkflow
from cellnition.science.networks_toolbox.netplot import plot_network

from cellnition.science.network_models.network_library import (StemCellNet,
                                                               StemCellTriad,
                                                               hESC_9a,
                                                               TrinodeCycle,
                                                               TrinodeCycleFullyConnected,
                                                               TrinodeCycleFullyConnected2,
                                                               StemCellTriadChain,
                                                              AKTNet,
                                                              BinodeCycle,
                                                              MAPK_net)

from cellnition.science.network_models.boolean_networks import BooleanNet
from cellnition.science.networks_toolbox.boolean_state_machine import BoolStateMachine
from cellnition.science.networks_toolbox.state_machine import StateMachine

# ....................{ MAIN                               }....................
# If this submodule is directly invoked by Python as the main module to be run
# (e.g., by being passed as the argument to the "-m" option), run this
# Streamlit-based web app.
# if __name__ == '__main__':


# ....................{ MAIN                               }....................
def main() -> None:
    '''
    Core function running this app: **Cellnition.**
    '''

    print("Starting the program")

    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    mono_edge = False
    fimg = '.png'

    # Save path for images and graphs:
    save_path_base = '/home/pietakio/Dropbox/Levin_2024/BooleanNetworks'
    save_path_date = os.path.join(save_path_base, 'Feb12_2025')
    if not os.path.isdir(save_path_date):
        os.makedirs(save_path_date)

    libg = MAPK_net()
    # libg = hESC_9a()

    # Specify how multiple interactions should combine:
    multi_coupling_type = CouplingType.mix1  # activators combine as "OR" and inhibitors "AND"
    # multi_coupling_type = CouplingType.additive # everything "OR"
    # multi_coupling_type = CouplingType.multiplicative # everything "AND"
    # multi_coupling_type = CouplingType.mix2  # activators combine "AND" and inhibitors "OR"

    constitutive_express = False  # activators present "AND" inhibitors absent for expression, when "False"

    verbose = False
    main_nodes_only = True

    graph_layout = 'dot'
    # graph_layout = 'neato'
    # graph_layout = 'circo'

    run_ind = 0

    print("Creating the Boolean model...")

    bn = BooleanNet()  # instantiate bool net solver
    bn.build_network_from_edges(libg.edges)  # build basic graph
    bn.characterize_graph()  # characterize the graph and set key params
    bn.set_node_types()

    bn.set_edge_types(libg.edge_types)  # set the edge types to the network

    # Save a plot of the graph:
    save_path_net = os.path.join(save_path_date, f'{libg.name}_{run_ind}')
    if not os.path.isdir(save_path_net):
        os.makedirs(save_path_net)

    graph_net_c = f'hier_graph_{libg.name}{fimg}'
    save_graph_net_hier = os.path.join(save_path_net, graph_net_c)

    cycle_tags = np.zeros(bn.N_nodes)
    cycle_tags[bn.nodes_in_cycles] = 1.0

    print("plotting networks")

    gp = plot_network(bn.nodes_list,
                      bn.edges_list,
                      bn.node_types,
                      bn.edge_types,
                      node_vals=bn.hier_node_level,
                      val_cmap='magma_r',
                      save_path=save_graph_net_hier,
                      layout='dot',
                      rev_font_color=False,
                      vminmax=(0.0, 1.0),
                      label_edges=False
                      )

    # Save a plot of the graph
    graph_net_c = f'circ_graph_{libg.name}{fimg}'
    save_graph_net_circo = os.path.join(save_path_net,
                                        graph_net_c)

    cycle_tags = np.zeros(bn.N_nodes)
    cycle_tags[bn.nodes_in_cycles] = 1.0

    gp = plot_network(bn.nodes_list,
                      bn.edges_list,
                      bn.node_types,
                      bn.edge_types,
                      node_vals=bn.hier_node_level,
                      val_cmap='magma_r',
                      save_path=save_graph_net_circo,
                      layout='circo',
                      rev_font_color=False,
                      vminmax=(0.0, 1.0),
                      label_edges=False
                      )

    # Save path for boolean net results:
    save_path_bool = os.path.join(save_path_net, f'bool_{multi_coupling_type.name[0:4]}')
    if not os.path.isdir(save_path_bool):
        os.makedirs(save_path_bool)

    c_vect_s, A_bool_s, A_bool_f = bn.build_boolean_model(use_node_name=True,
                                                          multi_coupling_type=multi_coupling_type,
                                                          constitutive_express=constitutive_express)

    sol_M, sol_char = bn.solve_system_equms(A_bool_f,
                                            constraint_inds=None,
                                            constraint_vals=None,
                                            signal_constr_vals=[0, 0, 1],
                                            search_main_nodes_only=True,
                                            n_max_steps=len(bn.main_nodes),
                                            verbose=False
                                            )

    # Save the solution states to a csv file:
    save_sols_data = os.path.join(save_path_bool, f'BoolSolM_{libg.name}.csv')
    np.savetxt(save_sols_data, sol_M[bn.noninput_node_inds, :], delimiter=',', header='')

# ....................{ MAIN ~ run                         }....................
# Run this app.
main()