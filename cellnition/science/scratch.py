
# def build_analytical_model_with_process(self,
#                                         control_edges_dict: dict|None = None,
#                                         control_node_dict: dict | None = None,
#                                         prob_acti: float=0.5,
#                                         edge_types: list|ndarray|None=None,
#                                         add_interactions: bool=True):
#     '''
#
#     '''
#
#     self._reduced_dims = False  # always build models in full dimensions
#
#     if edge_types is None:
#         self.edge_types = self.get_edge_types(p_acti=prob_acti)
#
#     else:
#         self.edge_types = edge_types
#
#     self.set_edge_types(edge_types, add_interactions)
#
#     # FIXME: we ultimately want to add more than one of each node type to the network (i.e. more
#     # than one effector, or multiple processes and sensors, etc)
#
#     if control_node_dict is None: # default behaviour
#         # Set indices of root and effector nodes to the highest and lowest degree nodes:
#         self._root_i = self.nodes_by_out_degree[0]
#         self._effector_i = self.nodes_by_out_degree[-1]
#         # add sensor and process nodes to the network as new nodes:
#         self._sensor_i = self.N_nodes
#         self._process_i = 1 + self.N_nodes
#
#         if control_edges_dict is None: # By default assign activator edge types
#             ps_edge = EdgeType.A.value
#             sr_edge = EdgeType.A.value
#
#         else: # allow user to specify
#             ps_edge = control_edges_dict['process-sensor'].value
#             sr_edge = control_edges_dict['sensor-root'].value
#
#         # Connect the new nodes to the network with new edges
#         self.GG.add_edge(self._process_i, self._sensor_i,
#                          edge_type=ps_edge)
#
#         self.GG.add_edge(self._sensor_i, self._root_i,
#                          edge_type=sr_edge)
#
#         # By default the effector-process edge type must be neutral as the
#         # process specifies the interaction of the effector on it:
#         self.GG.add_edge(self._effector_i, self._process_i,
#                          edge_type=EdgeType.N.value)
#
#         # update the graph nodes and node properties:
#         self.N_nodes = self.N_nodes + 2  # We added the sensor and process nodes to the graph
#         self.nodes_index = np.arange(self.N_nodes)  # We make a new nodes list
#
#         # Harvest data from edge attributes for edge property re-assignment:
#         self.read_edge_info_from_graph()
#
#         # Indices of key new edges:
#         self.ei_process_sensor = self.edges_index.index((self._process_i, self._sensor_i))
#         self.ei_sensor_root = self.edges_index.index((self._sensor_i, self._root_i))
#
#         # Re-calculate key characteristics of the graph after adding in new nodes and edges:
#         self.characterize_graph()
#
#     else:
#         self._root_i = control_node_dict['root']
#         self._effector_i = control_node_dict['effector']
#         self._sensor_i = control_node_dict['sensor']
#         self._process_i = control_node_dict['process']
#
#         # Indices of key new edges:
#         self.ei_process_sensor = self.edges_index.index((self._process_i, self._sensor_i))
#         self.ei_effector_process = self.edges_index.index((self._effector_i, self._process_i))
#
#         # Override the edge-type for the control loop effector-process:
#         self.edge_types[self.ei_effector_process] = EdgeType.N # This is always neutral
#
#         # Update the edge types on the graph edges:
#         self.set_edge_types(self.edge_types, add_interactions)
#
#     # See if there are paths connecting the hub and effector node:
#     try:
#         self.root_effector_paths = sorted(nx.shortest_simple_paths(self.GG, self._root_i, self._effector_i), reverse=True)
#     except:
#         self.root_effector_paths = []
#
#     # Now that indices are set, give nodes a type attribute:
#     node_types = [NodeType.gene for i in self.nodes_index]  # Set all nodes to the gene type
#
#     # Add a type tag to any nodes on the path between root hub and effector:
#     for path_i in self.root_effector_paths:
#         for nde_i in path_i:
#             node_types[nde_i] = NodeType.path
#
#     node_types[self._root_i] = NodeType.root  # Set the most connected node to the root hub
#     node_types[self._effector_i] = NodeType.effector  # Set the least out-connected node to the effector
#     node_types[self._sensor_i] = NodeType.sensor  # Set the sensor node
#     node_types[self._process_i] = NodeType.process  # Set the process node
#
#     # Set node types to the graph:
#     self.node_types = node_types
#     self.set_node_types(node_types)
#
#     # Build the basic edge functions:
#     self.edge_funcs = []
#     for et in self.edge_types:
#         if et is EdgeType.A:
#             self.edge_funcs.append(f_acti_s)
#         elif et is EdgeType.I:
#             self.edge_funcs.append(f_inhi_s)
#         elif et is EdgeType.N:
#             self.edge_funcs.append(f_neut_s)
#         else:
#             raise Exception("Edge type not found!")
#
#     # Rebuild the symbolic parameter bases:
#     c_s = sp.IndexedBase('c')
#     K_s = sp.IndexedBase('K')
#     n_s = sp.IndexedBase('n')
#     r_max_s = sp.IndexedBase('r_max')
#     d_max_s = sp.IndexedBase('d_max')
#
#     # # These are needed for lambdification of analytical models:
#     self.K_vect_s = [K_s[i] for i in range(self.N_edges)]
#     self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
#     self.r_vect_s = [r_max_s[i] for i in self.nodes_index]
#     self.d_vect_s = [d_max_s[i] for i in self.nodes_index]
#     self.c_vect_s = [c_s[i] for i in self.nodes_index]
#
#     # Create the analytic equations governing the process:
#     self.set_analytic_process(self.c_vect_s[self._effector_i], self.c_vect_s[self._process_i])
#
#     # Create the edge-function collections at each node for the GRN interactions:
#     efunc_vect = [[] for i in self.nodes_index]
#     for ei, ((i, j), fun_type) in enumerate(zip(self.edges_index, self.edge_funcs)):
#         efunc_vect[j].append(fun_type(self.c_vect_s[i], self.K_vect_s[ei], self.n_vect_s[ei]))
#
#     # Create the time-change vector with the process node math applied:
#     dcdt_vect_s = []
#
#     for nde_i, (fval_set, ntype) in enumerate(zip(efunc_vect, node_types)):
#         if ntype is NodeType.process:  # if we're dealing with the phys/chem process node...
#             dcdt_vect_s.append(self.dEdt_s)  # ...append the osmotic strain rate equation.
#
#         else:  # if it's any other kind of node insert the conventional GRN node dynamics
#             if add_interactions:
#                 if len(fval_set) == 0:
#                     normf = 1
#                 else:
#                     normf = sp.Rational(1, len(fval_set))
#
#                 dcdt_vect_s.append(self.r_vect_s[nde_i] * np.sum(fval_set) * normf -
#                                    self.c_vect_s[nde_i] * self.d_vect_s[nde_i])
#             else:
#                 dcdt_vect_s.append(self.r_vect_s[nde_i]*np.prod(fval_set) -
#                                    self.c_vect_s[nde_i]*self.d_vect_s[nde_i])
#
#
#     # The last thing we need to do is add on a rate term for those nodes that have no inputs,
#     # as they're otherwise ignored in the construction:
#     for nde_i, di in enumerate(self.in_degree_sequence):
#         if di == 0 and add_interactions is True:
#             dcdt_vect_s[nde_i] += self.r_vect_s[nde_i]
#
#     # analytical rate of change of concentration vector for the network:
#     self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)
#
#     self._include_process = True # Set the internal boolean to True for consistency
#     # Generate the optimization "energy" function as well as jacobians and hessians for the system:
#     self._generate_optimization_functions()


# def create_parameter_vects(self, Ki: float|list=0.5,
#                             ni:float|list=3.0,
#                             ri:float|list=1.0,
#                             di:float|list=1.0
#                             ):
#     '''
#     Create parameter vectors for the rate equations of the model nodes.
#     If floats are specified, use the same parameters for all edges and nodes in the network; if lists
#     are specified use them to create the model parameter vectors.
#
#     '''
#     # :
#     if type(Ki) is not list:
#         K_vect = []
#         for ei in range(self.N_edges):
#             K_vect.append(Ki)
#
#     else:
#         K_vect = Ki
#
#     if type(ni) is not list:
#         n_vect = []
#         for ei in range(self.N_edges):
#             n_vect.append(ni)
#     else:
#         n_vect = ni
#
#     if type(ri) is not list:
#         r_vect = []
#         for nde_i in range(self.N_nodes):
#             r_vect.append(ri)
#     else:
#         r_vect = ri
#
#     if type(di) is not list:
#         d_vect = []
#         for nde_i in range(self.N_nodes):
#             d_vect.append(di)
#     else:
#         d_vect = di
#
#     self.K_vect = K_vect
#     self.n_vect = n_vect
#     self.r_vect = r_vect
#     self.d_vect = d_vect


# def build_analytical_model(self,
#                            prob_acti: float=0.5,
#                            edge_types: list|ndarray|None=None,
#                            node_type_dict: dict|None=None,
#                            add_interactions: bool=True):
#     '''
#     Using the network to supply structure, this method constructs an
#     analytical (symbolic) model of the regulatory network as a dynamic
#     system.
#
#     prob_acti : float = 0.5
#         The probability of an edge being an activation. Note that this value
#         must be less than 1.0, and that the probability of an edge being an
#         inhibitor becomes 1.0 - p_acti.
#
#     edge_types : list|None
#         A list of edge type enumerations; one for each edge of the network. If edge_types
#         is not specified, then they will be randomly generated, with all self-loops set to be
#         activations.
#
#     node_type_dict : dict|None
#         Dictionary that allows for specification of different node types. If node names are
#         strings, the node dict can be used to assign a node type based on the first letter
#         of the node name. For example, if nodes labeled 'S0', 'S1', 'S2' are desired to be
#         signal-type nodes, then the node dictionary would specify {'S': NodeType.signal}.
#         Nodes that are numerically named must be individually specified by their number in the
#         dictionary. If node_type_dict is None, then all nodes become NodeType.Gene.
#
#     add_interactions : bool = False
#         In a network, the interaction of two or more regulators at a node can be multiplicative
#         (equivalent to an "And" condition) or additive (equivalent to an "or condition). This
#         bool specifies whether multiple interactions should be additive (True) or multiplicative (False).
#
#     '''
#
#     self._reduced_dims = False # always build models in full dimensions
#
#     if edge_types is None:
#         self.edge_types = self.get_edge_types(p_acti=prob_acti)
#
#     else:
#         self.edge_types = edge_types
#
#     self.set_edge_types(self.edge_types, add_interactions)
#
#     # Now that indices are set, give nodes a type attribute:
#     node_types = [NodeType.gene for i in self.nodes_index]  # First set all nodes
#     # to the gene type
#
#     if node_type_dict is not None:
#         for ntag, ntype in node_type_dict.items():
#             for nde_i, nde_n in enumerate(self.nodes_list):
#                 if type(nde_n) is str:
#                     if nde_n.startswith(ntag):
#                         node_types[nde_i] = ntype
#                 else:
#                     if nde_n == ntag:
#                         node_types[nde_i] = ntype
#
#     # Set node types to the graph:
#     self.node_types = node_types
#     self.set_node_types(node_types)
#
#     # Determine the node indices of any signal nodes:
#     self.signal_inds = []
#     for nde_i, nde_t in enumerate(self.node_types):
#         if nde_t is NodeType.signal:
#             self.signal_inds.append(nde_i)
#
#     self.nonsignal_inds = np.setdiff1d(self.nodes_index, self.signal_inds)
#
#     c_s = sp.IndexedBase('c')
#     K_s = sp.IndexedBase('K')
#     n_s = sp.IndexedBase('n')
#     r_max_s = sp.IndexedBase('r_max')
#     d_max_s = sp.IndexedBase('d_max')
#
#     # These are needed for lambdification of analytical models:
#     self.K_vect_s = [K_s[i] for i in range(self.N_edges)]
#     self.n_vect_s = [n_s[i] for i in range(self.N_edges)]
#     self.r_vect_s = [r_max_s[i] for i in self.nodes_index]
#     self.d_vect_s = [d_max_s[i] for i in self.nodes_index]
#     self.c_vect_s = [c_s[i] for i in self.nodes_index]
#
#     efunc_add_growthterm_vect = [[] for i in self.nodes_index]
#     efunc_mult_growthterm_vect = [[] for i in self.nodes_index]
#     efunc_mult_decayterm_vect = [[] for i in self.nodes_index]
#     for ei, ((i, j), fun_type, add_tag, gwth_tag) in enumerate(zip(self.edges_index,
#                                                 self.edge_funcs,
#                                                 self.add_edge_interaction_bools,
#                                                 self.growth_interaction_bools)):
#         if add_tag and gwth_tag:
#             efunc_add_growthterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
#             efunc_mult_growthterm_vect[j].append(None)
#             efunc_mult_decayterm_vect[j].append(None)
#
#         elif not add_tag and gwth_tag:
#             efunc_mult_growthterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
#             efunc_add_growthterm_vect[j].append(None)
#             efunc_mult_decayterm_vect[j].append(None)
#
#         elif not add_tag and not gwth_tag:
#             efunc_mult_decayterm_vect[j].append(fun_type(c_s[i], K_s[ei], n_s[ei]))
#             efunc_add_growthterm_vect[j].append(None)
#             efunc_mult_growthterm_vect[j].append(None)
#         else:
#             raise Exception("Currently not supporting any other node interaction types.")
#
#     self.efunc_add_growthterm_vect = efunc_add_growthterm_vect
#     self.efunc_mult_growthterm_vect = efunc_mult_growthterm_vect
#     self.efunc_mult_decayterm_vect = efunc_mult_decayterm_vect
#
#     dcdt_vect_s = []
#
#     # Process additive interactions acting on the growth term:
#     for nde_i, (fval_add, fval_mult, fval_multd) in enumerate(zip(efunc_add_growthterm_vect,
#                                                                efunc_mult_growthterm_vect,
#                                                                efunc_mult_decayterm_vect)):
#         if (np.all(np.asarray(fval_add) == None) and len(fval_add) != 0):
#             fsum = 1
#
#         elif len(fval_add) == 0:
#             fsum = 1
#
#         else:
#             fsum = 0
#             for fi in fval_add:
#                 if fi is not None:
#                     fsum += fi
#
#         # replace the segment in the efunc vect with the sum:
#         efunc_add_growthterm_vect[nde_i] = fsum
#
#     # # Process multiplicative interactions acting on the growth term:
#         fprodg = 1
#         for fi in fval_mult:
#             if fi is not None:
#                 fprodg = fprodg*fi
#
#         efunc_mult_growthterm_vect[nde_i] = fprodg
#
#     # Process multiplicative interactions acting on the decay term:
#         fprodd = 1
#         for fi in fval_multd:
#             if fi is not None:
#                 fprodd = fprodd*fi
#
#         efunc_mult_decayterm_vect[nde_i] = fprodd
#
#     # for ni in range(self.N_nodes): # Creating the sum terms above, construct the equation
#         ntype = self.node_types[nde_i]  # get the node type
#         # if we're not dealing with a 'signal' node that's written externally:
#         if ntype is not NodeType.signal:
#             dcdt_vect_s.append(r_max_s[nde_i]*efunc_mult_growthterm_vect[nde_i]*efunc_add_growthterm_vect[nde_i]
#                                - c_s[nde_i] * d_max_s[nde_i] * efunc_mult_decayterm_vect[nde_i])
#         else:
#             dcdt_vect_s.append(0)
#
#
#     # analytical rate of change of concentration vector for the network:
#     self.dcdt_vect_s = sp.Matrix(dcdt_vect_s)
#
#     self._include_process = False  # Set the internal boolean to True for consistency
#
#     # Generate the optimization "energy" function as well as jacobians and hessians for the system:
#     self._generate_optimization_functions()