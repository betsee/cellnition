#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines enumerations throughout Cellnition, primarily in the
construction of regulatory network models.
'''


from enum import Enum

class EdgeType(Enum):
    '''
    Specify whether the directed edge of the regulatory network
    has an activating effect on the level of the target node (`EdgeType.A`),
    an inhibiting effect on the level of the target node (`EdgeType.I`). In
    non-regulatory network models, an additional neutral directed
    edge `EdgeType.N` can be specified.

    '''
    A = 'Activator'
    I = 'Inhibitor'
    N = 'Neutral'
    As = 'Multiplicative Activation'
    Is = 'Multiplicative Inhibition'


class NodeType(Enum):
    '''
    Specify the node type of the model. Typically, nodes are
    `NodeType.gene` by default. Other node types are ascribed for
    tagging nodes in networks (e.g. `NodeType.Cycle` could be
    used to color nodes present in network cycles), but generally
    NodeTypes other than `NodeType.gene` are not presently utilized
    in Cellnition and are planned for future work.

    '''
    gene = 'Gene'
    signal = 'Signal'
    process = 'Process'
    sensor = 'Sensor'
    effector = 'Effector'
    core = 'Hub Core'
    factor = 'Factor'
    cycle = 'Cycle'

class GraphType(Enum):
    '''
    When creating procedural graphs, the `GraphType.scale_free`
    tag can be used to generate graphs with scale-free degree
    distributions, whereas the `GraphType.random` can be used to
    generate random networks with binomial degree distributions.

    '''
    scale_free = 'Scale Free'
    random = 'Random'
    user = 'User Defined'

class EquilibriumType(Enum):
    '''
    When treating the regulatory network as a dynamic system, the
    identified equilibrium states have a particular dynamic character,
    which Cellnition marks using EquilibriumType:

    - `attractor`: an asymptotically-stable point attractor that moves monotonically to the attractor.
    - `attractor_limit_cycle`: an asymptotically-stable attractor that moves with diminishing-amplitude oscillations to the attractor.
    - `limit_cycle`: an attractor that perpetually repeats cyclic oscillations about a central point.
    - `saddle`: a metastable saddle-type attractor that leaves the attractor with small perturbations.
    - `undetermined`: an equilibrium with undetermined characteristics.
    - `hidden`: an attractor that was only found in time-series or pseudo-time series investigations.

    '''
    attractor = 0
    attractor_limit_cycle = 1
    limit_cycle = 2
    saddle = 3
    repellor = 4
    repellor_limit_cycle = 5
    undetermined = 6
    hidden = 7 # corresponds to a hidden attractor

class InterFuncType(Enum):
    '''
    In continuous models (see
    [`ProbabilityNet`][cellnition.science.network_models.probability_networks.ProbabilityNet]),
    this enumeration specifies the type of interaction function used when the level of one node
    acts to regulate the level of another node.
    InterFuncType.logistic specifies logistic function interactions
    (see [`f_acti_logi_s`][cellnition.science.network_models.interaction_functions] and
    [`f_inhi_logi_s`][cellnition.science.network_models.interaction_functions.f_inhi_logi_s]), whereas
    InterFuncType.hill specifies hill function interactions (see
    [`f_acti_hill_s`][cellnition.science.network_models.interaction_functions.f_acti_hill_s] and
    [`f_inhi_hill_s`][cellnition.science.network_models.interaction_functions.f_inhi_hill_s]).

    '''
    logistic = 'Logistic'
    hill = 'Hill'

class CouplingType(Enum):
    '''
    In both continuous ([`ProbabilityNet`][cellnition.science.network_models.probability_networks.ProbabilityNet])
    and Boolean ([`BooleanNet`][cellnition.science.network_models.boolean_networks.BooleanNet]) regulatory
    network models, the CouplingType supplies a heuristic for the case where multiple nodes regulate the activity
    of a single downstream node. The different coupling types are:

    - `additive`: when multiple nodes act on a downstream node, their influences combine additively ("OR" function),
    regardless of whether they are activators or inhibitors.
    - `multiplicative`: when multiple nodes act on a downstream node, their influences combine multiplicatively ("AND" function),
    regardless of whether they are activators or inhibitors.
    - `mix1`: when multiple nodes act on a downstream node, activators combine additively ("OR" function), while
     inhibitors combine multiplicatively ("AND" function).
    - `mix2`: when multiple nodes act on a downstream node, inhibitors combine additively ("OR" function), while
     activators combine multiplicatively ("AND" function).

    '''
    additive = 'additive'
    multiplicative = 'multiplicative'
    mix1 = 'mix1' # activators "OR", inhibitors "AND"
    specified = 'specified'
    mix2 = 'mix2' # Activators "AND", inhibitors "OR"
