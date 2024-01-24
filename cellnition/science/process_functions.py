#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module builds the model network as a symbolic graph, has attributes to
analyze the network, and has the ability to build an analytic (symbolic) model
that can be used to study the network as a continuous dynamic system.

Note on model parameterization:
For a general regulatory network, one can say the rate of change of agent a_i is:
d a_i/dt = r_max*sum(f(a_j)) - a_i*d_max
Where d_max is maximum rate of decay, r_max is maximum rate of growth, and f(a_j) is
an interaction function detailing how ajent a_j influences the growth of a_i.

Here we use normalized agent variables: c_i = a_i/alpha with alpha = (r_i/d_i).
We use the substitution, a_i = c_i*alpha for all entities in the network rate equations.
Then we note that if we're using Hill equations, then for each edge with index ei and
node index i acting on node j we can define an additional parameter,
beta_ei = r_i/(K_ei*d_i) where K_ei is the Hill coefficient for the edge interaction, and
r_i and d_i are the maximum rate of growth and decay (respectively) for node i acting on j
via edge ei.

The result is an equation, which at steady-state is only dependent on the parameters beta_ei and
the Hill exponent n_ei. In kinetics, the node d_i multiplies through the equation to define a
relative rate of change, however, in steady-state searches this d_i multiplies out (assuming d_i != 0).
'''
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from scipy.optimize import minimize, fsolve
from scipy.signal import square
import networkx as nx
from networkx import DiGraph
import sympy as sp
from sympy.core.symbol import Symbol
from sympy.tensor.indexed import Indexed
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
from cellnition.science.interaction_functions import f_acti_s, f_inhi_s, f_neut_s
import pygraphviz as pgv
