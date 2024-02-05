# #!/usr/bin/env python3
# # --------------------( LICENSE                           )--------------------
# # Copyright (c) 2023-2024 Alexis Pietak.
# # See "LICENSE" for further details.
#
# '''
# This module builds the model network as a symbolic graph, has attributes to
# analyze the network, and has the ability to build an analytic (symbolic) model
# that can be used to study the network as a continuous dynamic system.
#
# Note on model parameterization:
# For a general regulatory network, one can say the rate of change of agent a_i is:
# d a_i/dt = r_max*sum(f(a_j)) - a_i*d_max
# Where d_max is maximum rate of decay, r_max is maximum rate of growth, and f(a_j) is
# an interaction function detailing how ajent a_j influences the growth of a_i.
#
# Here we use normalized agent variables: c_i = a_i/alpha with alpha = (r_i/d_i).
# We use the substitution, a_i = c_i*alpha for all entities in the network rate equations.
# Then we note that if we're using Hill equations, then for each edge with index ei and
# node index i acting on node j we can define an additional parameter,
# beta_ei = r_i/(K_ei*d_i) where K_ei is the Hill coefficient for the edge interaction, and
# r_i and d_i are the maximum rate of growth and decay (respectively) for node i acting on j
# via edge ei.
#
# The result is an equation, which at steady-state is only dependent on the parameters beta_ei and
# the Hill exponent n_ei. In kinetics, the node d_i multiplies through the equation to define a
# relative rate of change, however, in steady-state searches this d_i multiplies out (assuming d_i != 0).
# '''
# import csv
# import numpy as np
# from numpy import ndarray
# from scipy.optimize import minimize, fsolve
# import sympy as sp
# from sympy.core.symbol import Symbol
# from sympy.tensor.indexed import Indexed
# from cellnition.science.network_models.network_abc import NetworkABC
# from cellnition.science.network_models.network_enums import GraphType, NodeType, InterFuncType
# from cellnition.science.network_models.interaction_functions import (f_acti_hill_s,
#                                                                      f_inhi_hill_s,
#                                                                      f_acti_logi_s,
#                                                                      f_inhi_logi_s)
#
# # TODO: Add in stochasticity

# class GeneNetworkModel(NetworkABC):
#     '''
#     This class allows one to generate a network using a random construction
#     algorithm, or from user-input edges. It then performs analysis on the
#     resulting graph to determine cycles, input and output degree distributions,
#     and hierarchical attributes. The class then enables the user to build an
#     analytical (i.e. symbolic math) module using the network, and has various routines
#     to determine equilibrium points and stable states of the network model. The
#     class then allows for time simulation of the network model as a dynamic system.
#
#     Public Attributes
#     -----------------
#     GG = nx.DiGraph(self.edges_list)
#     N_nodes = N_nodes # number of nodes in the network (as defined by user initially)
#     nodes_index = self.nodes_list
#     nodes_list = sorted(self.GG.nodes())
#     N_edges = len(self.edges_list)
#     edges_index = self.edges_list
#     edges_list
#
# # Indices of edges with selfloops:
#         self.selfloop_edge_inds = [self.edges_list.index(ei) for ei in list(nx.selfloop_edges(self.GG))]
#
#         # Degree analysis:
#         self.in_degree_sequence = [deg_i for nde_i, deg_i in
#                                    self.GG.in_degree(self.nodes_list)] # aligns with node order
#
#         self.in_dmax = np.max(self.in_degree_sequence)
#
#
#         self.out_degree_sequence = [deg_i for nde_i, deg_i in
#                                     self.GG.out_degree(self.nodes_list)]  # aligns with node order
#
#         # The outward flow of interaction at each node of the graph:
#         self.node_divergence = np.asarray(self.out_degree_sequence) - np.asarray(self.in_degree_sequence)
#
#         self.out_dmax = np.max(self.out_degree_sequence)
#         self.in_dmax = np.max(self.in_degree_sequence)
#
#         self.in_bins, self.in_degree_counts = np.unique(self.in_degree_sequence,
#                                                         return_counts=True)
#         self.out_bins, self.out_degree_counts = np.unique(self.out_degree_sequence,
#                                                           return_counts=True)
#
#         # Nodes sorted by number of out-degree edges:
#         self.nodes_by_out_degree = np.flip(np.argsort(self.out_degree_sequence))
#
#         self.nodes_by_in_degree = np.flip(np.argsort(self.in_degree_sequence))
#
#         self.root_hub = self.nodes_by_out_degree[0]
#         self.leaf_hub = self.nodes_by_out_degree[-1]
#
#         # Number of cycles:
#         self.graph_cycles = sorted(nx.simple_cycles(self.GG))
#         self.N_cycles = len(self.graph_cycles)
#
#             dcdt_vect_reduced_s = None
#
#         self.nodes_in_cycles = list(nodes_in_cycles)
#         self.nodes_acyclic = np.setdiff1d(self.nodes_index, nodes_in_cycles)
#
#                     self.hier_node_level = np.zeros(self.N_nodes)
#             self.hier_incoherence = 0.0
#             self.dem_coeff = 0.0
#
#     '''
#
#     def __init__(self,
#                  N_nodes: int,
#                  edges: list|ndarray|None = None,
#                  graph_type: GraphType = GraphType.user,
#                  b_param: float = 0.20,
#                  g_param: float=0.75,
#                  delta_in: float=0.0,
#                  delta_out: float = 0.0,
#                  p_edge: float=0.2,
#                  interaction_function_type: InterFuncType = InterFuncType.hill
#                  ):
#         '''
#         Initialize the class and build and characterize a network.
#
#         Parameters
#         -----------
#         N_nodes: int
#             The number of nodes to build the network (only used in randomly built networks, otherwise the number of
#             nodes is calculated from the number of unique nodes supplied in the edges list).
#
#         edges: list|ndarray|None = None
#             A list of tuples that defines edges of a network, where each directed edge is a pair of nodes. The
#             nodes may be integers or strings, but cannot be mixed type. If edges is left as None, then a
#             graph will be randomly constructed.
#
#         graph_type: GraphType = GraphType.scale_free
#             The type of graph to generate in randomly-constructed networks.
#
#         b_param: float = 0.20
#             For scale-free randomly-constructed networks, this determines the amount of interconnectivity between
#             the in and out degree distributions, and in practical terms, increases the number of cycles in the graph.
#             Note that 1 - beta - gamma must be greater than 0.0.
#
#         g_param: float=0.75
#             For scale-free randomly-constructed networks, this determines the emphasis on the network's
#             out degree distribution, and in practical terms, increases the scale-free character of the out-distribution
#             of the graph. Note that 1 - beta - gamma must be greater than 0.0.
#
#         delta_in: float=0.0
#             A parameter that increases the complexity of the network core, leading to more nodes being involved in
#             cycles.
#         delta_out: float = 0.0
#             A parameter that increases the complexity of the network core, leading to more nodes being involved in
#             cycles.
#
#         p_edge: float=0.2
#             For randomly constructed binomial-type networks, this parameter determines the probability of forming
#             an edge. As p_edge increases, the number of network edges increases drammatically.
#
#
#
#         '''
#
#         super().__init__(N_nodes) # Initialize the base class
#
#         self._inter_fun_type = interaction_function_type
#
#         if interaction_function_type is InterFuncType.hill:
#             # build using Hill equations:
#             self._f_acti_s = f_acti_hill_s
#             self._f_inhi_s = f_inhi_hill_s
#
#         else:
#             # build using Logistic equations:
#             self._f_acti_s = f_acti_logi_s
#             self._f_inhi_s = f_inhi_logi_s
#
#         self._N_nodes = N_nodes # number of nodes in the network (as defined by user initially)
#
#         # Depending on whether edges are supplied by user, generate
#         # a graph:
#         if edges is None:
#             self.randomly_generate_special_network(b_param=b_param,
#                                                    g_param=g_param,
#                                                    graph_type=graph_type,
#                                                    delta_in=delta_in,
#                                                    delta_out=delta_out,
#                                                    p_edge=p_edge)
#
#             self.edges_index = self.edges_list
#             self.nodes_index = self.nodes_list
#
#         else:
#             self.build_network_from_edges(edges)
#
#         # Calculate key characteristics of the graph
#         self.characterize_graph()
#
#     def build_analytical_model(self,
#                                prob_acti: float=0.5,
#                                edge_types: list|ndarray|None=None,
#                                node_type_dict: dict|None=None,
#                                add_interactions: bool=True,
#                                pure_gene_edges_only: bool=False):
#         '''
#         Using the network to supply structure, this method constructs an
#         analytical (symbolic) model of the regulatory network as a dynamic
#         system.
#
#         prob_acti : float = 0.5
#             The probability of an edge being an activation. Note that this value
#             must be less than 1.0, and that the probability of an edge being an
#             inhibitor becomes 1.0 - p_acti.
#
#         edge_types : list|None
#             A list of edge type enumerations; one for each edge of the network. If edge_types
#             is not specified, then they will be randomly generated, with all self-loops set to be
#             activations.
#
#         node_type_dict : dict|None
#             Dictionary that allows for specification of different node types. If node names are
#             strings, the node dict can be used to assign a node type based on the first letter
#             of the node name. For example, if nodes labeled 'S0', 'S1', 'S2' are desired to be
#             signal-type nodes, then the node dictionary would specify {'S': NodeType.signal}.
#             Nodes that are numerically named must be individually specified by their number in the
#             dictionary. If node_type_dict is None, then all nodes become NodeType.Gene.
#
#         add_interactions : bool = False
#             In a network, the interaction of two or more regulators at a node can be multiplicative
#             (equivalent to an "And" condition) or additive (equivalent to an "or condition). This
#             bool specifies whether multiple interactions should be additive (True) or multiplicative (False).
#
#         pure_gene_edges_only : bool = False
#             In a network with mixed node types, setting this to true will ensure that only edges
#             that connect two nodes that are both NodeType.gene will be included in the model. If
#             it's set to "False", then NodeType.gene, NodeType.sensor, and NodeType.effector are all
#             included as valid edges.
#
#         '''
#
#         self._reduced_dims = False # always build models in full dimensions
#
#         if edge_types is None:
#             self.edge_types = self.get_edge_types(p_acti=prob_acti)
#
#         else:
#             self.edge_types = edge_types
#
#         self.set_edge_types(self.edge_types, add_interactions)
#
#         # Set node types to the graph:
#         self.set_node_types(node_type_dict=node_type_dict, pure_gene_edges_only=pure_gene_edges_only)
#
#         # Next we want to distinguish a subset of edges that connect only "regular nodes":
#         self.regular_edges_index = []
#         for ei, (nde_i, nde_j) in enumerate(self.edges_index):
#             if nde_i in self.regular_node_inds and nde_j in self.regular_node_inds:
#                 self.regular_edges_index.append((nde_i, nde_j))
#
#         # Next, create a process dict that organizes features required for
#         # constructing the heterogenous process:
#         self.processes_list = []
#         if len(self.process_node_inds):
#             for pi in self.process_node_inds:
#                 process_dict = {NodeType.process.name: pi} # store the process node
#                 # Allow for multiple effectors, sensors and factors:
#                 process_dict[NodeType.effector.name] = []
#                 process_dict[NodeType.sensor.name] = []
#                 process_dict[NodeType.factor.name] = []
#                 # set through all edges:
#                 for ei, (nde_i, nde_j) in enumerate(self.edges_index):
#                     # Find an edge that connects to this process
#                     if pi == nde_i:
#                         process_dict[self.node_types[nde_j].name].append(nde_j)
#                     elif pi == nde_j:
#                         process_dict[self.node_types[nde_i].name].append(nde_i)
#
#                 self.processes_list.append(process_dict)
#
#         B_s = sp.IndexedBase('beta') # Symbolic base Hill 'beta parameter' edge interaction
#         n_s = sp.IndexedBase('n') # Symbolic base Hill exponent for each edge interaction
#         d_max_s = sp.IndexedBase('d_max') # Symbolic base decay rate for each node
#
#         # These are needed for lambdification of analytical models:
#         self.B_vect_s = [B_s[i] for i in range(self.N_edges)]
#         self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
#         self.d_vect_s = [d_max_s[i] for i in self.nodes_index]
#
#         if type(self.nodes_list[0]) is str:
#             # If nodes have string names, let the math reflect these:
#             self._c_vect_s = [sp.symbols(ndi) for ndi in self.nodes_list]
#         else:
#             # Otherwise, if nodes are just numbered, let it be a generic 'c' symbol
#             # to node number:
#             c_s = sp.IndexedBase('c')
#             self._c_vect_s = [c_s[i] for i in self.nodes_index]
#
#         # If there are sensors in the network, described by logistic function
#         # they have their own parameters:
#         if len(self.sensor_node_inds):
#             k_s = sp.IndexedBase('k') # steepness of sensor reaction
#             co_s = sp.IndexedBase('co') # centre of sensor reaction
#             self.sensor_params_s = {}
#             for ii, sens_i in enumerate(self.sensor_node_inds):
#                 self.sensor_params_s[sens_i] = (co_s[ii], k_s[ii])
#         else:
#             self.sensor_params_s = {}
#
#         efunc_add_growthterm_vect = [[] for i in self.nodes_index]
#         efunc_mult_growthterm_vect = [[] for i in self.nodes_index]
#         efunc_mult_decayterm_vect = [[] for i in self.nodes_index]
#
#         for ei, ((i, j), fun_type, add_tag, gwth_tag) in enumerate(zip(self.edges_index,
#                                                     self.edge_funcs,
#                                                     self.add_edge_interaction_bools,
#                                                     self.growth_interaction_bools)):
#             if (i, j) in self.regular_edges_index:
#                 if add_tag and gwth_tag:
#                     efunc_add_growthterm_vect[j].append(fun_type(self._c_vect_s[i], B_s[ei], n_s[ei]))
#                     efunc_mult_growthterm_vect[j].append(None)
#                     efunc_mult_decayterm_vect[j].append(None)
#
#                 elif not add_tag and gwth_tag:
#                     efunc_mult_growthterm_vect[j].append(fun_type(self._c_vect_s[i], B_s[ei], n_s[ei]))
#                     efunc_add_growthterm_vect[j].append(None)
#                     efunc_mult_decayterm_vect[j].append(None)
#
#                 elif not add_tag and not gwth_tag:
#                     efunc_mult_decayterm_vect[j].append(fun_type(self._c_vect_s[i], B_s[ei], n_s[ei]))
#                     efunc_add_growthterm_vect[j].append(None)
#                     efunc_mult_growthterm_vect[j].append(None)
#                 else:
#                     raise Exception("Currently not supporting any other node interaction types.")
#
#         self.efunc_add_growthterm_vect = efunc_add_growthterm_vect
#         self.efunc_mult_growthterm_vect = efunc_mult_growthterm_vect
#         self.efunc_mult_decayterm_vect = efunc_mult_decayterm_vect
#
#         # Initialize the dcdt_vect_s with zeros:
#         dcdt_vect_s = [0 for nni in self.nodes_index]  # equations for rate of change of a node
#         eq_c_vect_s = [0 for nni in self.nodes_index]  # null equations for direct solutions for a node
#
#         # Process additive interactions acting on the growth term:
#         for nde_i, (fval_add, fval_mult, fval_multd) in enumerate(zip(efunc_add_growthterm_vect,
#                                                                    efunc_mult_growthterm_vect,
#                                                                    efunc_mult_decayterm_vect)):
#             if (np.all(np.asarray(fval_add) == None) and len(fval_add) != 0):
#                 fsum = 1
#
#             elif len(fval_add) == 0:
#                 fsum = 1
#
#             else:
#                 fsum = 0
#                 for fi in fval_add:
#                     if fi is not None:
#                         fsum += fi
#
#             # replace the segment in the efunc vect with the sum:
#             efunc_add_growthterm_vect[nde_i] = fsum
#
#         # # Process multiplicative interactions acting on the growth term:
#             fprodg = 1
#             for fi in fval_mult:
#                 if fi is not None:
#                     fprodg = fprodg*fi
#
#             efunc_mult_growthterm_vect[nde_i] = fprodg
#
#         # Process multiplicative interactions acting on the decay term:
#             fprodd = 1
#             for fi in fval_multd:
#                 if fi is not None:
#                     fprodd = fprodd*fi
#
#             efunc_mult_decayterm_vect[nde_i] = fprodd
#
#             # for ni in range(self.N_nodes): # Creating the sum terms above, construct the equation
#             # if we're not dealing with a 'special' node that follows non-Hill dynamics:
#             if (nde_i in self.node_type_inds[NodeType.gene.name] or
#                     nde_i in self.node_type_inds[NodeType.effector.name]):
#                 dcdt_vect_s[nde_i] = (d_max_s[nde_i] * efunc_mult_growthterm_vect[nde_i] * efunc_add_growthterm_vect[nde_i]
#                                       - self._c_vect_s[nde_i] * d_max_s[nde_i] * efunc_mult_decayterm_vect[nde_i])
#             elif nde_i in self.process_node_inds:
#                 proc_i = self.process_node_inds.index(nde_i) # get the index of the process
#                 proc_dict = self.processes_list[proc_i] # get the correct info dictionary for the process
#                 i_eff = proc_dict[NodeType.effector.name] # allow for multiple effectors
#                 i_fact = proc_dict[NodeType.factor.name][0]
#                 i_proc = proc_dict[NodeType.process.name]
#                 i_sens = proc_dict[NodeType.sensor.name][0]
#
#                 c_proc = self._c_vect_s[i_proc]
#                 c_fact = self._c_vect_s[i_fact]
#
#                 if len(i_eff) == 1:
#                     c_eff = self._c_vect_s[i_eff[0]]
#                     # Add the osmotic process attributes to the network:
#                     self.osmotic_process(c_eff, c_proc, c_fact)
#
#                 elif len(i_eff) == 2:
#                     c_eff0 = self._c_vect_s[i_eff[0]]
#                     c_eff1 = self._c_vect_s[i_eff[1]]
#                     self.osmotic_process(c_eff0, c_proc, c_fact, c_channel=c_eff1)
#
#                 else:
#                     raise Exception("To define a process must have either 1 or 2 effectors!")
#
#                 # and add the process rate equation to the change vector:
#                 dcdt_vect_s[nde_i] = self.dEdt_s
#
#                 # Finally, we need an 'auxillary' equation that represents the value of the
#                 # sensor in terms of the process, but equals zero (so we can include it in
#                 # optimization root finding):
#                 eq_c_vect_s[i_sens] = -self._c_vect_s[i_sens] + f_acti_logi_s(c_proc,
#                                                                               self.sensor_params_s[i_sens][0],
#                                                                               self.sensor_params_s[i_sens][1])
#
#         # construct an 'extra params' list that holds non-GRN parameter arguments
#         # from sensors and the osmotic process:
#         self.extra_params_s = []
#
#         for k, v in self.sensor_params_s.items():
#             self.extra_params_s.extend(v)
#
#         self.extra_params_s.extend(self.process_params_s)
#
#         self.eq_c_vect_s = sp.Matrix(eq_c_vect_s)
#
#         # analytical rate of change of concentration vector for the network:
#         self._dcdt_vect_s = sp.Matrix(dcdt_vect_s)
#
#         self._include_process = False  # Set the internal boolean to True for consistency
#
#         # vector to find roots of in any search (want to find the zeros of this function):
#         self._dcdt_vect_s = self.eq_c_vect_s + self._dcdt_vect_s
#
#         # Generate the optimization "energy" function as well as jacobians and hessians for the system:
#         self._generate_optimization_functions()
#
#     def _generate_optimization_functions(self):
#         '''
#         Using the model equations, generate numerical optimization functions
#         and rate functions required to solve system properties numerically.
#         '''
#
#         if self._reduced_dims and self._solved_analytically is False:
#             dcdt_vect_s = self._dcdt_vect_reduced_s
#             c_vect_s = self._c_vect_reduced_s
#
#         else:
#             dcdt_vect_s = self._dcdt_vect_s
#             c_vect_s = self._c_vect_s
#
#         lambda_params = self._fetch_lambda_params_s(self._c_vect_s)
#         lambda_params_r = self._fetch_lambda_params_s(c_vect_s)
#
#         # Create a Jacobian for the whole system
#         self.jac_s = self._dcdt_vect_s.jacobian(sp.Matrix(self._c_vect_s))
#
#         # The Hessian is a more complex tensor for the whole system:
#         self.hess_s = sp.Array(
#             [[[self._dcdt_vect_s[i].diff(dcj).diff(dci) for dcj in self._c_vect_s]
#               for dci in self._c_vect_s] for i in range(self._N_nodes)])
#
#         # Optimization function for solving the problem (defined in terms of reduced dimensions):
#         self.opti_s = (dcdt_vect_s.T*dcdt_vect_s)[0]
#         self.opti_jac_s = sp.Array([self.opti_s.diff(ci) for ci in c_vect_s])
#         self.opti_hess_s = sp.Matrix(self.opti_jac_s).jacobian(c_vect_s)
#
#         # Lambdify the two outputs so they can be used to study the network numerically:
#         # On the whole system:
#         flatten_f = np.asarray([fs for fs in self._dcdt_vect_s])
#         self.dcdt_vect_f = sp.lambdify(lambda_params, flatten_f)
#         self.jac_f = sp.lambdify(lambda_params, self.jac_s)
#         self.hess_f = sp.lambdify(lambda_params, self.hess_s)
#
#         # These will automatically become the correct dimensions due to definition of lambda_params_r:
#         self.opti_f = sp.lambdify(lambda_params_r, self.opti_s)
#         self.opti_jac_f = sp.lambdify(lambda_params_r, self.opti_jac_s)
#         self.opti_hess_f = sp.lambdify(lambda_params_r, self.opti_hess_s)
#
#         # For case of reduced dims, we need two additional attributes lambdified:
#         # If dims are reduced we also need to lambdify the remaining concentration sets
#         if self._reduced_dims and self._solved_analytically is False:
#             # This is now the same thing as opti-f:
#             flatten_fr = np.asarray([fs for fs in self._dcdt_vect_reduced_s])
#             self.dcdt_vect_reduced_f = sp.lambdify(lambda_params_r, flatten_fr)
#
#             # Create a reduced Jacobian:
#             self.jac_reduced_s = self._dcdt_vect_reduced_s.jacobian(sp.Matrix(self._c_vect_reduced_s))
#             self.jac_reduced_f = sp.lambdify(lambda_params_r, self.jac_reduced_s)
#
#             self.sol_cset_f = {}
#             for ci, eqci in self.sol_cset_s.items():
#                 c_ind = self._c_vect_s.index(ci)
#                 self.sol_cset_f[c_ind] = sp.lambdify(lambda_params_r, eqci)
#
#     def generate_split_optimization_functions(self, constrained_node_inds: int | list):
#         '''
#         Generates numerical optimization functions that are split such that nodes identified
#         in signal_node_inds become function arguments with values set by the user,
#         whereas the rest of the nodes are solved for.
#
#         '''
#
#         self._constrained_node_inds = constrained_node_inds
#         nonconstrained_node_inds = np.setdiff1d(self.nodes_index, constrained_node_inds).tolist()
#         self._nonconstrained_node_inds = nonconstrained_node_inds
#
#         self.c_vect_nosigs_s = np.asarray(self._c_vect_s)[nonconstrained_node_inds].tolist()
#         self.c_vect_sigs_s = np.asarray(self._c_vect_s)[constrained_node_inds].tolist()
#         self.dcdt_vect_nosigs_s = self._dcdt_vect_s[nonconstrained_node_inds, :]
#
#         # obtain symbolic parameters to work with the split system:
#         lambda_params = self._fetch_lambda_params_s(self.c_vect_nosigs_s, self.c_vect_sigs_s)
#
#         # Re-lambdify the change vector with the new parameter arrangements:
#         # flatten_f = np.asarray([fs for fs in self.dcdt_vect_s])
#         flatten_f = np.asarray([fs for fs in self.dcdt_vect_nosigs_s])
#         self.dcdt_vect_nosigs_f = sp.lambdify(lambda_params, flatten_f)
#
#         # Optimization function for solving the problem (defined in terms of reduced dimensions):
#         # opti_s = (self.dcdt_vect_s.T*self.dcdt_vect_s)[0]
#         opti_s = (self.dcdt_vect_nosigs_s.T * self.dcdt_vect_nosigs_s)[0]
#         opti_jac_s = sp.Array([opti_s.diff(ci) for ci in self.c_vect_nosigs_s])
#         opti_hess_s = sp.Matrix(opti_jac_s).jacobian(self.c_vect_nosigs_s)
#
#         self.opti_nosigs_f = sp.lambdify(lambda_params, opti_s)
#         self.opti_jac_nosigs_f = sp.lambdify(lambda_params, opti_jac_s)
#         self.opti_hess_nosigs_f = sp.lambdify(lambda_params, opti_hess_s)
#
#     def create_parameter_vects(self, beta_base: float | list=2.0,
#                                n_base: float | list=3.0,
#                                d_base: float | list=1.0,
#                                co: float|list = 0.0,
#                                ki: float|list = 10.0
#                                ):
#         '''
#         Create parameter vectors for the rate equations of the model nodes.
#         If floats are specified, use the same parameters for all edges and nodes in the network; if lists
#         are specified use them to create the model parameter vectors.
#
#         '''
#         # :
#         if type(beta_base) is not list:
#             B_vect = []
#             for ei in range(self.N_edges):
#                 B_vect.append(beta_base)
#
#         else:
#             B_vect = beta_base
#
#         if type(n_base) is not list:
#             n_vect = []
#             for ei in range(self.N_edges):
#                 n_vect.append(n_base)
#         else:
#             n_vect = n_base
#
#         if type(d_base) is not list:
#             d_vect = []
#             for nde_i in range(self._N_nodes):
#                 d_vect.append(d_base)
#         else:
#             d_vect = d_base
#
#         if type(ki) is not list:
#             k_vect = []
#             for nde_i in self.sensor_node_inds:
#                 k_vect.append(ki)
#
#         else:
#             k_vect = ki
#
#         if type(co) is not list:
#             co_vect = []
#             for nde_i in self.sensor_node_inds:
#                 co_vect.append(co)
#
#         else:
#             co_vect = co
#
#         self.beta_vect = B_vect
#         self.n_vect = n_vect
#         self.d_vect = d_vect
#
#         self.extra_params_f = []
#         for cc, kk in zip(co_vect, k_vect):
#             self.extra_params_f.extend((cc, kk))
#
#         if len(self.process_params_s):
#             self.extra_params_f.extend(self.process_params_f)
#
#     def _fetch_lambda_params_s(self, cvect_s: list, cko_s: Symbol|Indexed|list|None = None) -> tuple:
#         '''
#         Obtain the correct set of parameters to use
#         when creating numerical lambda functions from
#         sympy analytical functions.
#         '''
#         if cko_s is None:
#             lambda_params = [cvect_s,
#                                self.d_vect_s,
#                                self.B_vect_s,
#                                self.n_vect_s]
#         else:
#             lambda_params = [cvect_s,
#                              cko_s,
#                                self.d_vect_s,
#                                self.B_vect_s,
#                                self.n_vect_s]
#
#         if len(self.process_node_inds) or len(self.sensor_node_inds):
#             lambda_params.append(self.extra_params_s)
#
#         return tuple(lambda_params)
#
#     def _fetch_function_args_f(self, cko: float|list|None = None) -> tuple:
#         '''
#         Obtain the correct set of auxiliary parameters (arguments) to use
#         when evaluating numerical lambda functions created
#         from sympy analytical functions in optimization functions.
#         '''
#         if cko is None:
#             lambda_params = [self.d_vect,
#                              self.beta_vect,
#                              self.n_vect]
#         else:
#             lambda_params = [cko,
#                              self.d_vect,
#                              self.beta_vect,
#                              self.n_vect]
#
#         if len(self.process_node_inds) or len(self.sensor_node_inds):
#             lambda_params.append(self.extra_params_f)
#
#         return tuple(lambda_params)
#
#     def generate_state_space(self,
#                              c_vect_s: list|ndarray,
#                              Ns: int=3,
#                              cmin: float=0.0,
#                              cmax: float=1.0,
#                              include_signals: bool = False  # include signal node states in the search?
#                              ):
#         '''
#
#         '''
#         c_test_lin_set = []
#
#         if self._reduced_dims and self._solved_analytically is False:
#             signal_inds = self.signal_reduced_inds
#             nonsignal_inds = self.nonsignal_reduced_inds
#         else:
#             signal_inds = self.input_node_inds
#             nonsignal_inds = self.noninput_node_inds
#
#         if include_signals is False:
#             # Create a c_vect sampled to the non-signal nodes:
#             c_vect = np.asarray(c_vect_s)[nonsignal_inds].tolist()
#
#         else:
#             c_vect = c_vect_s # otherwise use the whole vector
#
#         for nd_i, ci in enumerate(c_vect):
#             i = c_vect.index(ci)
#             if i in self.process_node_inds:
#                 c_test_lin_set.append(np.linspace(self.strain_min, self.strain_max, Ns))
#             else:
#                 c_test_lin_set.append(np.linspace(cmin, cmax, Ns))
#
#         # Create a set of matrices specifying the concentration grid for each
#         # node of the network:
#         C_test_M_SET = np.meshgrid(*c_test_lin_set, indexing='ij')
#
#         # Create linearized arrays for each concentration, stacked into one column per node:
#         c_test_set = np.asarray([cM.ravel() for cM in C_test_M_SET]).T
#
#         if include_signals is False:  # then we need to add on a zeros block for signal state
#             n_rows_test = c_test_set.shape[0]
#             signal_block = np.zeros((n_rows_test, len(signal_inds)))
#             c_test_set = np.column_stack((c_test_set, signal_block))
#
#         return c_test_set, C_test_M_SET, c_test_lin_set
#
#
#     def optimized_phase_space_search(self,
#                                      Ns: int=3,
#                                      cmin: float=0.0,
#                                      cmax: float=1.0,
#                                      c_bounds: list|None = None,
#                                      tol:float = 1.0e-15,
#                                      round_sol: int=6, # decimals to round solutions to prevent duplicates
#                                      method: str='Root', # Solve by finding the roots of the dc/dt equation
#                                      include_signals: bool = False  # include signal node states in the search?
#                                      ):
#         '''
#
#         '''
#
#         if self.dcdt_vect_f is None:
#             raise Exception("Must use the method build_analytical_model to generate attributes"
#                             "to use this function.")
#
#         # Determine the set of additional arguments to the optimization function:
#         function_args = self._fetch_function_args_f()
#
#         # Initialize the equillibrium point solutions to be a set:
#         mins_found = []
#
#         # If it's already been solved analytically, we can simply plug in the variables to obtain the solution
#         # at the minimum rate:
#         if self._solved_analytically:
#             mins_foundo = [[] for i in range(self._N_nodes)]
#             for ii, eqi in self.sol_cset_f.items():
#                 mins_foundo[ii] = eqi(*function_args)
#
#             mins_found.append(mins_foundo)
#
#         else: # if we don't have an explicit solution:
#             # Otherwise, we need to go through the whole optimization:
#             if self._reduced_dims:
#                 N_nodes = len(self._c_vect_reduced_s)
#                 c_vect_s = self._c_vect_reduced_s
#                 dcdt_funk = self.dcdt_vect_reduced_f
#             else:
#                 N_nodes = self._N_nodes
#                 c_vect_s = self._c_vect_s
#                 dcdt_funk = self.dcdt_vect_f
#
#             # Generate the points in state space to sample at:
#             c_test_set, _, _ = self.generate_state_space(c_vect_s,
#                                                          Ns=Ns,
#                                                          cmin=cmin,
#                                                          cmax=cmax,
#                                                          include_signals=include_signals)
#
#             if c_bounds is None:
#                 c_bounds = [(cmin, cmax) for i in range(N_nodes)]
#
#                 if len(self.process_node_inds):
#                     if self._reduced_dims is False:
#                         for ip in self.process_node_inds:
#                             c_bounds[ip] = (self.strain_min, self.strain_max)
#                 else:
#                     for ip in self.process_node_inds:
#                         if self._c_vect_s[ip] in self._c_vect_reduced_s:
#                             i = self._c_vect_reduced_s.index(self._c_vect_s[ip])
#                             c_bounds[i] = (self.strain_min, self.strain_max)
#
#             self._c_bounds = c_bounds # save for inspection later
#
#             for c_vecti in c_test_set:
#                 if method == 'Powell' or method == 'trust-constr':
#                     if method == 'Powell':
#                         jac = None
#                         hess = None
#                     else:
#                         jac = self.opti_jac_f
#                         hess = self.opti_hess_f
#
#                     sol0 = minimize(self.opti_f,
#                                     c_vecti,
#                                     args=function_args,
#                                     method=method,
#                                     jac=jac,
#                                     hess=hess,
#                                     bounds=c_bounds,
#                                     tol=tol,
#                                     callback=None,
#                                     options=None)
#
#                     mins_found.append(sol0.x)
#
#                 else:
#                     sol_root = fsolve(dcdt_funk, c_vecti, args=function_args, xtol=tol)
#                     if len(self.process_node_inds) == 0:
#                         # If we're not using the process, constrain all concs to be above zero
#                         if (np.all(np.asarray(sol_root) >= 0.0)):
#                             mins_found.append(sol_root)
#                     else:
#                         # get the nodes that must be constrained above zero:
#                         conc_nodes = np.setdiff1d(self.nodes_index, self.process_node_inds)
#                         # Then, only the nodes that are gene products must be above zero
#                         if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
#                             mins_found.append(sol_root)
#
#             if self._reduced_dims is False:
#                 self.mins_found = mins_found
#
#             else: # we need to piece together the full solution as the minimum will only be a subset of all
#                 # concentrations
#                 full_mins_found = []
#                 for mins_foundi in list(mins_found): # for each set of unique minima found
#                     mins_foundo = [[] for i in range(self._N_nodes)]
#                     for cmi, mi in zip(self.c_master_i, mins_foundi):
#                         for ii, eqi in self.sol_cset_f.items():
#                             mins_foundo[ii] = eqi(mins_foundi, *function_args) # compute the sol for this conc.
#                         # also add-in the minima for the master concentrations to the full list
#                         mins_foundo[cmi] = mi
#                     # We've redefined the mins list so it now includes the full set of concentrations;
#                     # flatten the list and add it to the new set:
#                     full_mins_found.append(mins_foundo)
#
#                 # Redefine the mins_found set for the full concentrations
#                 mins_found = full_mins_found
#
#         # ensure the list is unique:
#         mins_found = np.round(mins_found, round_sol)
#         mins_found = np.unique(mins_found, axis=0).tolist()
#
#         return mins_found
#
#     def constrained_phase_space_search(self,
#                                        c_vals: list[float], # values of nodes that are constrained
#                                        c_inds: list[int], # indices of nodes that are constrained
#                                        Ns: int=3,
#                                        cmin: float=0.0,
#                                        cmax: float=1.0,
#                                        c_bounds: list|None = None,
#                                        tol:float = 1.0e-15,
#                                        round_sol: int=6, # decimals to round solutions to prevent duplicates
#                                        method: str='Root', # Solve by finding the roots of the dc/dt equation
#                                      ):
#         '''
#         Search the phase space for equilibrium solutions while using
#         simple constraints on sensors and factors that are input by
#         the user and substituted into the split optimization function
#         as arguments rather than variables.
#
#         '''
#
#         if self.dcdt_vect_f is None:
#             raise Exception("Must use the method build_analytical_model to generate attributes"
#                             "to use this function.")
#
#         # generate new optimization functions that have the ability to set fixed values to c_inds:
#         self.generate_split_optimization_functions(c_inds)
#
#         # Determine the set of additional arguments to the optimization function:
#         function_args = self._fetch_function_args_f(cko=c_vals)
#
#         # Initialize the equillibrium point solutions to be stored:
#         mins_found = []
#
#         # Otherwise, we need to go through the whole optimization:
#         N_nodes = len(self.c_vect_nosigs_s)
#         c_vect_s = self.c_vect_nosigs_s
#         dcdt_funk = self.dcdt_vect_nosigs_f
#
#         # Generate the points in state space to sample at:
#         c_test_set, _, _ = self.generate_state_space(c_vect_s,
#                                                      Ns=Ns,
#                                                      cmin=cmin,
#                                                      cmax=cmax,
#                                                      include_signals=True)
#
#         if c_bounds is None:
#             c_bounds = [(cmin, cmax) for i in range(N_nodes)]
#
#             if len(self.process_node_inds):
#                 for ip in self.process_node_inds:
#                     c_bounds[ip] = (self.strain_min, self.strain_max)
#
#         self._c_bounds = c_bounds # save for inspection later
#
#         for c_vecti in c_test_set:
#             if method == 'Powell' or method == 'trust-constr':
#                 if method == 'Powell':
#                     jac = None
#                     hess = None
#                 else:
#                     jac = self.opti_jac_nosigs_f
#                     hess = self.opti_hess_nosigs_f
#
#                 sol0 = minimize(self.opti_nosigs_f,
#                                 c_vecti,
#                                 args=function_args,
#                                 method=method,
#                                 jac=jac,
#                                 hess=hess,
#                                 bounds=c_bounds,
#                                 tol=tol,
#                                 callback=None,
#                                 options=None)
#
#                 mins_found.append(sol0.x)
#
#             else:
#                 sol_root = fsolve(dcdt_funk, c_vecti, args=function_args, xtol=tol)
#                 if len(self.process_node_inds) == 0:
#                     # If we're not using the process, constrain all concs to be above zero
#                     if (np.all(np.asarray(sol_root) >= 0.0)):
#                         mins_found.append(sol_root)
#                 else:
#                     # get the nodes that must be constrained above zero:
#                     conc_nodes = np.setdiff1d(self.nodes_index, self.process_node_inds)
#                     # Then, only the nodes that are gene products must be above zero
#                     if (np.all(np.asarray(sol_root)[conc_nodes] >= 0.0)):
#                         mins_found.append(sol_root)
#
#                 # Next we need to rebuild the system to include the constrained node
#                 # values in the full solution vector:
#
#         # ensure the list is unique:
#         mins_found = np.round(mins_found, round_sol)
#         mins_found = np.unique(mins_found, axis=0).tolist()
#
#         # Finally, need to add in the values of the constrained nodes to create
#         # a full solution vector:
#         MM = np.zeros((len(mins_found), self._N_nodes))
#         MM[:, self._nonconstrained_node_inds] = mins_found
#         MM[:, self._constrained_node_inds] = c_vals
#
#         return MM.tolist()
#
#     def stability_estimate(self,
#                            mins_found: set|list,
#                            ):
#         '''
#
#         '''
#
#         eps = 1.0e-25 # we need a small value to add to avoid dividing by zero
#
#         sol_dicts_list = []
#         # in some Jacobians
#         for cminso in mins_found:
#
#             solution_dict = {}
#
#             # print(f'min vals: {cminso}')
#             solution_dict['Minima Values'] = cminso
#
#             cmins = np.asarray(cminso) + eps # add the small amount here, before calculating the jacobian
#
#             func_args = self._fetch_function_args_f()
#
#             # print(f'dcdt at min: {self.dcdt_vect_f(cmins, r_vect, d_vect, K_vect, n_vect)}')
#
#             solution_dict['Change at Minima'] = self.dcdt_vect_f(cmins, *func_args)
#
#             jac = self.jac_f(cmins, *func_args)
#
#            # get the eigenvalues of the jacobian at this equillibrium point:
#             eig_valso, eig_vects = np.linalg.eig(jac)
#
#             # round the eigenvalues so we don't have issue with small imaginary components
#             eig_vals = np.round(np.real(eig_valso), 1) + np.round(np.imag(eig_valso), 1)*1j
#
#             # print(f'Jacobian eigs: {eig_vals}')
#
#             solution_dict['Jacobian Eigenvalues'] = eig_vals
#
#             # get the indices of eigenvalues that have only real components:
#             real_eig_inds = (np.imag(eig_vals) == 0.0).nonzero()[0]
#             # print(real_eig_inds)
#
#             # If all eigenvalues are real and they're all negative:
#             if len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) <= 0.0):
#                 # print('Stable Attractor')
#                 char_tag = 'Stable Attractor'
#
#             # If all eigenvalues are real and they're all positive:
#             elif len(real_eig_inds) == len(eig_vals) and np.all(np.real(eig_vals) > 0.0):
#                 # print('Stable Repellor')
#                 char_tag = 'Stable Repellor'
#
#             # If there are no real eigenvalues we only know its a limit cycle but can't say
#             # anything certain about stability:
#             elif len(real_eig_inds) == 0 and np.all(np.real(eig_vals) <= 0.0):
#                 # print('Stable Limit cycle')
#                 char_tag = 'Stable Limit Cycle'
#
#             # If there are no real eigenvalues and a mix of real component sign, we only know its a limit cycle but can't say
#             # anything certain about stability:
#             elif len(real_eig_inds) == 0 and np.any(np.real(eig_vals) > 0.0):
#                 # print('Limit cycle')
#                 char_tag = 'Limit Cycle'
#
#             elif np.all(np.real(eig_vals[real_eig_inds]) <= 0.0):
#                 # print('Stable Limit Cycle')
#                 char_tag = 'Stable Limit Cycle'
#
#             elif np.any(np.real(eig_vals[real_eig_inds] > 0.0)):
#                 # print('Saddle Point')
#                 char_tag = 'Saddle Point'
#             else:
#                 # print('Undetermined Stability Status')
#                 char_tag = 'Undetermined'
#
#             solution_dict['Stability Characteristic'] = char_tag
#
#             sol_dicts_list.append(solution_dict)
#
#             # print('----')
#         self.sol_dicts_list = sol_dicts_list
#
#         return sol_dicts_list
#
#     def run_time_sim(self, tend: float,
#                      dt: float,
#                      cvecti: ndarray|list,
#                      sig_inds: ndarray|list|None = None,
#                      sig_times: ndarray | list | None = None,
#                      sig_mag: ndarray | list | None = None,
#                      dt_samp: float|None = None,
#                      constrained_inds: list | None = None,
#                      constrained_vals: list | None = None,
#                      d_base: float = 1.0,
#                      n_base: float = 3.0,
#                      beta_base: float = 4.0
#                      ):
#         '''
#
#         '''
#         Nt = int(tend/dt)
#         tvect = np.linspace(0.0, tend, Nt)
#
#         if sig_inds is not None:
#             c_signals = self.make_signals_matrix(tvect, sig_inds, sig_times, sig_mag)
#         else:
#             c_signals = None
#
#         concs_time = []
#
#         # sampling compression
#         if dt_samp is not None:
#             sampr = int(dt_samp / dt)
#             tvect_samp = tvect[0::sampr]
#             tvectr = tvect_samp
#         else:
#             tvect_samp = None
#             tvectr = tvect
#
#         # make a time-step update vector so we can update any sensors as
#         # an absolute reading (dt = 1.0) while treating the kinetics of the
#         # other node types:
#         dtv = 1.0e-3 * np.ones(self._N_nodes)
#         dtv[self.sensor_node_inds] = 1.0
#
#         function_args = self._fetch_function_args_f()
#
#         for ti, tt in enumerate(tvect):
#             dcdt = self.dcdt_vect_f(cvecti, *function_args)
#             cvecti += dtv * dcdt
#
#             if c_signals is not None:
#                 # manually set the signal node values:
#                 cvecti[self.input_node_inds] = c_signals[ti, self.input_node_inds]
#
#             if dt_samp is None:
#                 concs_time.append(cvecti * 1)
#             else:
#                 if tt in tvect_samp:
#                     concs_time.append(cvecti * 1)
#
#         concs_time = np.asarray(concs_time)
#
#         return concs_time, tvectr
#
#     def find_attractor_sols(self,
#                             sols_0: list|ndarray,
#                             tol: float=1.0e-3,
#                             verbose: bool=True,
#                             N_round: int = 12,
#                             unique_sols: bool = False,
#                             sol_round: int = 1,
#                             save_file: str|None = None
#                             ):
#         '''
#
#         '''
#
#         sol_char_0 = self.stability_estimate(sols_0)
#
#         solsM = []
#         sol_char_list = []
#         sol_char_error = []
#         i = 0
#         for sol_dic in sol_char_0:
#             error = np.sum(sol_dic['Change at Minima']**2)
#             char = sol_dic['Stability Characteristic']
#             sols = sol_dic['Minima Values']
#
#             if char != 'Saddle Point' and error <= tol:
#                 i += 1
#                 if verbose and unique_sols is False:
#                     print(f'Soln {i}, {char}, {sols}, {np.round(error, N_round)}')
#                 solsM.append(sols)
#                 sol_char_list.append(char)
#                 sol_char_error.append(error)
#
#         solsM_return = np.asarray(solsM).T
#
#         if unique_sols and len(solsM) != 0:
#             # round the sols to avoid degenerates and return indices to the unique solutions:
#             solsy, inds_solsy = np.unique(np.round(solsM, sol_round), axis=0, return_index=True)
#             if verbose:
#                 for i, si in enumerate(inds_solsy):
#                     print(f'Soln {i}: {sol_char_list[si]}, {solsM[si]}, error: {sol_char_error[si]}')
#
#             solsM_return = np.asarray(solsM)[inds_solsy].T
#
#         if save_file is not None:
#             solsMi = np.asarray(solsM)
#             header = [f'State {i}' for i in range(solsMi.shape[0])]
#             with open(save_file, 'w', newline="") as file:
#                 csvwriter = csv.writer(file)  # create a csvwriter object
#                 csvwriter.writerow(header)  # write the header
#                 csvwriter.writerow(sol_char_error)  # write the root error at steady-state
#                 csvwriter.writerow(sol_char_list)  # write the attractor characterization
#                 for si in solsMi.T:
#                     csvwriter.writerow(si)  # write the soln data rows for each gene
#
#         return solsM_return








