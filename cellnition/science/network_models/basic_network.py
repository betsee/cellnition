#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module allows one to build a network without an analytical model and analyze the
network in terms of graph theory.

'''
import csv
from collections.abc import Callable
import numpy as np
from numpy import ndarray
from scipy.optimize import fsolve
import sympy as sp
from sympy import MutableDenseMatrix
from cellnition.science.network_models.network_abc import NetworkABC
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             InterFuncType,
                                                             CouplingType,
                                                             EquilibriumType)

class BasicNet(NetworkABC):
    '''

    '''

    def __init__(self,
                 N_nodes: int,
):
        '''

        '''

        super().__init__(N_nodes)  # Initialize the base class

    # def build_network_from_edges(self,
    #                              net_edges: list[tuple],
    #                              count_cycles: bool=True,
    #                              cycle_length_bound: int|None=None):
    #
    #     self.build_network_from_edges(net_edges)
    #     self.characterize_graph(count_cycles=count_cycles,
    #                             cycle_length_bound=cycle_length_bound)
    #
    # def randomly_generate_special_network(self, b_param: float = 0.15,
    #                                       g_param: float = 0.8,
    #                                       delta_in: float = 0.0,
    #                                       delta_out: float = 0.0,
    #                                       p_edge: float = 0.5,
    #                                       graph_type: GraphType = GraphType.scale_free,
    #                                       count_cycles: bool = True,
    #                                       cycle_length_bound: int | None = None
    #                                       ):
    #
    #     self.randomly_generate_special_network(b_param=b_param,
    #                                            g_param= g_param,
    #                                            delta_in = delta_in,
    #                                            delta_out = delta_out,
    #                                            p_edge=p_edge,
    #                                            graph_type=graph_type
    #                                            )
    #
    #     self.characterize_graph(count_cycles=count_cycles,
    #                             cycle_length_bound=cycle_length_bound)
