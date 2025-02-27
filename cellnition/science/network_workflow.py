#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines a top-level handler that can perform various workflows pertaining to network generation,
analytis, model generation, solution finding, searching, knockout experiments, and other functions.
'''
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             InterFuncType,
                                                             CouplingType,
                                                             PType)
from cellnition.science.network_models.probability_networks import ProbabilityNet
from cellnition.science.networks_toolbox.netplot import plot_network
from cellnition.science.networks_toolbox.gene_knockout import GeneKnockout
from cellnition.science.networks_toolbox.phase_space_searches import multistability_search
from cellnition.science.networks_toolbox.state_machine import StateMachine
from cellnition.science.network_models.basic_network import BasicNet
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata


# FIXME: document throughout

class NetworkWorkflow(object):
    '''

    '''
    def __init__(self, save_path: str):
        '''

        '''
        self._save_path = save_path
        
        
    def bionet_graph_gen(self, 
                         N_nodes: int,
                         p_in_type: PType,
                         p_out_type: PType,
                         connect_fract: float=0.25,
                         p_bkg: float=0.002,
                         p_activator: float=0.5
                         ):
        '''
        
        '''
        N_iter = int(connect_fract * (N_nodes * N_nodes))

        nodes_list = [i for i in range(N_nodes)]
        node_act_list = np.random.choice([True, False], size=N_nodes, p=[p_activator, 1.0 - p_activator])
        in_degree_list = np.zeros(N_nodes)
        out_degree_list = np.zeros(N_nodes)

        edges_list = [(0, 1)]  # Initial edges list
        edge_types_list = [EdgeType.A] # Initial edge type list

        out_degree_list[0] = 1
        in_degree_list[1] = 1

        p_rand_list = np.asarray([1.0 / N_nodes for i in nodes_list])  # random probability

        for i in range(N_iter):

            ave_degree_list = (np.asarray(in_degree_list) +
                               np.asarray(out_degree_list)) / 2  # calculate ave of in and out degree

            p_in_list = (np.asarray(in_degree_list) + p_bkg) / np.sum(np.asarray(in_degree_list) + p_bkg)
            p_out_list = (np.asarray(out_degree_list) + p_bkg) / np.sum(np.asarray(out_degree_list) + p_bkg)
            p_ave_list = (np.asarray(ave_degree_list) + p_bkg) / np.sum(np.asarray(ave_degree_list) + p_bkg)
            
            if p_in_type is PType.in_deg:
                p_in = p_in_list
            elif p_in_type is PType.out_deg:
                p_in = p_out_list
            elif p_in_type is PType.ave_deg:
                p_in = p_ave_list
            else:
                p_in = p_rand_list

            if p_out_type is PType.in_deg:
                p_out = p_in_list
            elif p_out_type is PType.out_deg:
                p_out = p_out_list
            elif p_out_type is PType.ave_deg:
                p_out = p_ave_list
            else:
                p_out = p_rand_list

            # print(p_out)
            # print(p_in)

            # select a node and connector from list of all nodes:
            node_a = np.random.choice(nodes_list, p=p_out)
            node_b = np.random.choice(nodes_list, p=p_in)

            if (node_a, node_b) not in edges_list:
                edges_list.append((node_a, node_b))

                # Determine if node_a is an activator or inhibitor:
                node_a_type = node_act_list[node_a]
                if node_a_type:
                    edge_types_list.append(EdgeType.A)
                else:
                    edge_types_list.append(EdgeType.I)

        return edges_list, edge_types_list

    def scalefree_graph_gen(self,
                            N_nodes: int,
                            b_param: float,
                            g_param: float,
                            delta_in: float,
                            delta_out: float,
                            i: int,
                            interaction_function_type: InterFuncType = InterFuncType.logistic,
                            coupling_type: CouplingType = CouplingType.mix1):
        '''

        '''
        a_param = 1.0 - b_param - g_param # calculate the a-parameter for sf net gen
        # Initialize an instance of probability nets:
        pnet = ProbabilityNet(N_nodes, interaction_function_type=interaction_function_type)
        # randomly generate a scale-free network model:
        pnet.randomly_generate_special_network(b_param=b_param,
                                               g_param=g_param,
                                               delta_in=delta_in,
                                               delta_out=delta_out,
                                               graph_type= GraphType.scale_free)
        # characterize the network:
        pnet.characterize_graph()

        # randomly generate edge types:
        edge_types = pnet.get_edge_types()

        # set the edge and node types to the network:
        pnet.set_edge_types(edge_types)
        pnet.set_node_types()

        # Get the signed adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                                              coupling_type=coupling_type)
        # Build the analytical model
        pnet.build_analytical_model(A_add_s, A_mul_s)

        dem_coeff = np.round(pnet.dem_coeff, 1)
        incoh = np.round(pnet.hier_incoherence, 1)
        fname_base = f'{i}_sf{N_nodes}_b{b_param}_g{g_param}_Ncycles{pnet.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: params {np.round(a_param,2), b_param, g_param, delta_in, delta_out}, '
                         f'cycles: {pnet.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoh.: {incoh}')

        return pnet, update_string, fname_base

    def binomial_graph_gen(self,
                           N_nodes: int,
                           p_edge: float,
                           i: int,
                           interaction_function_type: InterFuncType = InterFuncType.logistic,
                           coupling_type: CouplingType = CouplingType.mix1
                           ):
        '''

        '''
        # Initialize an instance of probability nets:
        pnet = ProbabilityNet(N_nodes, interaction_function_type=interaction_function_type)
        # randomly generate a scale-free network model:
        pnet.randomly_generate_special_network(p_edge=p_edge,
                                               graph_type=GraphType.random)
        # characterize the network:
        pnet.characterize_graph()

        # randomly generate edge types:
        edge_types = pnet.get_edge_types()

        # set the edge and node types to the network:
        pnet.set_edge_types(edge_types)
        pnet.set_node_types()

        # Get the signed adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                                              coupling_type=coupling_type)
        # Build the analytical model
        pnet.build_analytical_model(A_add_s, A_mul_s)

        dem_coeff = np.round(pnet.dem_coeff, 1)
        incoh = np.round(pnet.hier_incoherence, 1)
        fname_base = f'{i}_bino{N_nodes}_Ncycles{pnet.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: params {p_edge}, '
                         f'cycles: {pnet.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoherence: {incoh}')

        return pnet, update_string, fname_base

    def make_network_from_edges(self,
                                  edges: list[tuple],
                                  edge_types: list[EdgeType]|None = None,
                                  node_type_dict: dict | None = None,
                                  interaction_function_type: InterFuncType=InterFuncType.logistic,
                                  coupling_type: CouplingType=CouplingType.mix1,
                                  network_name: str='network',
                                  i: int=0,
                                  verbose: bool=False,
                                  build_analytical_model: bool=True,
                                  count_cycles: bool=True,
                                  cycle_length_bound: int|None=None):
        '''

        '''

        if verbose:
            print("Begining network build...")

        N_nodes = np.unique(np.ravel(edges)).shape[0]
        pnet = ProbabilityNet(N_nodes, interaction_function_type=interaction_function_type)
        if verbose:
            print("Building network...")
        pnet.build_network_from_edges(edges)

        if verbose:
            print("Characterizing network...")
        # characterize the network:
        pnet.characterize_graph(count_cycles=count_cycles,
                                cycle_length_bound=cycle_length_bound)

        if edge_types is None:
            # randomly generate edge types:
            edge_types = pnet.get_edge_types()

        if verbose:
            print("Setting edge types network...")
        # set the edge and node types to the network:
        pnet.set_edge_types(edge_types)
        pnet.set_node_types(node_type_dict=node_type_dict)

        if verbose:
            print("Building adjacency matrix...")

        # Get the adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                          coupling_type=coupling_type)
        if build_analytical_model:
            if verbose:
                print("Building analytical model...")
            # build the analytical model for this network:
            pnet.build_analytical_model(A_add_s, A_mul_s)

        fname_base = f'{i}_{network_name}'

        dem_coeff = np.round(pnet.dem_coeff, 1)
        incoh = np.round(pnet.hier_incoherence, 1)

        if count_cycles is False:
            pnet.N_cycles = 9999

        update_string = (f'{i}: cycles: {pnet.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoherence: {incoh}')

        if verbose:
            print("Completed network build!")

        return pnet, update_string, fname_base

    def read_graph_from_file(self,
                             filename: str,
                             interaction_function_type: InterFuncType = InterFuncType.logistic,
                             coupling_type: CouplingType = CouplingType.mix1,
                             i: int=0):
        '''
        Read a network, including edge types, from a saved file.

        '''
        GG = nx.read_gml(filename, label=None)
        nodes_list = sorted(GG.nodes())
        N_nodes = len(nodes_list)

        edges_list = []
        edge_types = []

        # get data stored on edge type key:
        edge_data = nx.get_edge_attributes(GG, "edge_type")

        for ei, et in edge_data.items():
            # append the edge to the list:
            edges_list.append(ei)
            edge_types.append(EdgeType[et])

        node_types = []

        # get data stored on node type key:
        node_data = nx.get_node_attributes(GG, "node_type")

        node_type_dict = {}

        for nde_i, nde_t in node_data.items():
            if type(nde_i) == str:
                node_type_dict[nde_i[0]] = NodeType[nde_t]
            else:
                node_type_dict[nde_i] = NodeType[nde_t]

            node_types.append(NodeType[nde_t])

        # Build a gene network with the properties read from the file:
        pnet = ProbabilityNet(N_nodes, interaction_function_type=interaction_function_type)
        pnet.build_network_from_edges(edges_list)

        # characterize the network:
        pnet.characterize_graph()

        pnet.set_edge_types(edge_types)
        # Assign node types to the network model:
        pnet.set_node_types(node_type_dict=node_type_dict)

        # Get the adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                          coupling_type=coupling_type)
        # build the analytical model for this network:
        pnet.build_analytical_model(A_add_s, A_mul_s)

        dem_coeff = np.round(pnet.dem_coeff, 1)
        incoh = np.round(pnet.hier_incoherence, 1)

        fname_base = f'{i}_bino{N_nodes}_Ncycles{pnet.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: cycles: {pnet.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoherence: {incoh}')

        return pnet, update_string, fname_base

    def bionet_work_frame(self,
                          N_nodes: int,
                          p_in_type: PType,
                          p_out_type: PType,
                          connect_fract: float=0.25,
                          p_bkg: float=0.002,
                          p_act: float=0.5,
                          interfunctype: InterFuncType=InterFuncType.logistic,
                          coupling_type: CouplingType=CouplingType.mix1,
                          fname_base: str='network',
                          frame_i: int=0,
                          save_path: str|None = None,
                          verbose: bool=True,
                          img_fmt: str='png',
                          N_input_nodes: int=3):
        '''

        '''



        edges_list, edge_types = self.bionet_graph_gen(N_nodes,
                                                          p_in_type,
                                                          p_out_type,
                                                          connect_fract,
                                                          p_bkg,
                                                          p_act
                                                          )

        pnet, update_string, net_name = self.make_network_from_edges(edges_list,
                                                                     edge_types=edge_types,
                                                                     # node_type_dict=node_type_dict,
                                                                     interaction_function_type=interfunctype,
                                                                     coupling_type=coupling_type,
                                                                     network_name=fname_base,
                                                                     i=frame_i,
                                                                     build_analytical_model=True,
                                                                     count_cycles=True,
                                                                     cycle_length_bound=10)

        if save_path is None:
            save_path = os.path.join(self._save_path, net_name)
        else:
            save_path = os.path.join(self._save_path, save_path)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # print key data:
        if verbose:
            print(f'N_nodes: {pnet.N_nodes} \n '
                  f'N_cycles: {pnet.N_cycles} \n'
                  f'N_edges: {pnet.N_edges} \n'
                  f'Connectivity: {np.round(pnet.N_edges / pnet.N_nodes ** 2, 4)} \n'
                  f'Hierarchical incoherence: {np.round(pnet.hier_incoherence, 4)} \n'
                  )

        # Save a plot of the graph
        graph_net_c = f'hier_graph_{net_name}.{img_fmt}'
        save_graph_net_hier = os.path.join(save_path, graph_net_c)

        cycle_tags = np.zeros(pnet.N_nodes)
        cycle_tags[pnet.nodes_in_cycles] = 1.0

        gp = plot_network(pnet.nodes_list,
                          pnet.edges_list,
                          pnet.node_types,
                          pnet.edge_types,
                          node_vals=pnet.hier_node_level,
                          # val_cmap = 'magma',
                          save_path=save_graph_net_hier,
                          layout='dot',
                          rev_font_color=False,
                          vminmax=(0.0, 1.0),
                          label_edges=False
                          )

        # Save a plot of the graph
        graph_net_c = f'circ_graph_{net_name}.{img_fmt}'
        save_graph_net_circo = os.path.join(save_path,
                                            graph_net_c)

        cycle_tags = np.zeros(pnet.N_nodes)
        cycle_tags[pnet.nodes_in_cycles] = 1.0

        gp = plot_network(pnet.nodes_list,
                          pnet.edges_list,
                          pnet.node_types,
                          pnet.edge_types,
                          # node_vals = cycle_tags,
                          # val_cmap = 'Blues',
                          save_path=save_graph_net_circo,
                          layout='circo',
                          rev_font_color=False,
                          vminmax=(0.0, 1.0),
                          label_edges=False
                          )

        fig, ax = pnet.plot_degree_distributions()
        deg_dist_fig = os.path.join(save_path, f'degree_dist{net_name}.{img_fmt}')
        plt.savefig(deg_dist_fig, dpi=300, transparent=True, format=img_fmt)
        plt.close(fig)

        plt.figure(figsize=(10, 10))
        deg_cor_fig = os.path.join(save_path, f'degree_correlation{net_name}.{img_fmt}')
        plt.scatter(pnet.in_degree_sequence, pnet.out_degree_sequence, c='k')
        for i, nde_nme in enumerate(pnet.nodes_list):
            xi = pnet.in_degree_sequence[i]
            yi = pnet.out_degree_sequence[i]
            plt.text(xi, yi, nde_nme)
        plt.xlabel('In degree')
        plt.xlabel('Out degree')
        plt.savefig(deg_cor_fig, dpi=300, transparent=True, format=img_fmt)
        plt.close()

        if verbose:
            print(f'Original input inds: {pnet.input_node_inds}')
        i_hi_sort = np.argsort(pnet.hier_node_level)

        if len(pnet.input_node_inds) == 0:
            if verbose:
                print('No input node inds; choosing top from hierarchical node level....')
            pnet.input_node_inds = i_hi_sort[0:N_input_nodes].tolist()

        pnet.noninput_node_inds = np.setdiff1d(pnet.nodes_index, pnet.input_node_inds).tolist()


        # # save the randomly generated network as a text file:
        # gfile = f'network_{net_name}.gml'
        # save_gfile = os.path.join(save_path, gfile)
        # pnet.save_network(save_gfile)

        dd = 1.0
        d_base = [dd for i in range(pnet.N_nodes)]

        if interfunctype is InterFuncType.logistic:
            n_base = 15.0
            bb = 0.5
            beta_base = [bb for i in range(pnet.N_edges)]
        else:
            n_base = 3.0
            bb = 2.0
            beta_base = [bb for i in range(pnet.N_edges)]

        smach = StateMachine(pnet)
        N_round_sol = 1
        return_saddles = False

        if verbose:
            print('Obtaining steady-state solutions...')

        solsM_all, charM_all, sols_list, states_dict, sig_test_set = smach.steady_state_solutions_search(
            beta_base=beta_base,
            n_base=n_base,
            d_base=d_base,
            verbose=verbose,
            return_saddles=return_saddles,
            N_space=2,
            search_tol=1.0e-15,
            sol_tol=1.0e-3,
            N_round_sol=N_round_sol,
            search_main_nodes_only=False,
            cluster_threshhold=0.1,
            cluster_method='inconsistent'
            )

        if verbose:
            print(f'Number of unique states found: {solsM_all.shape[1]}')
        # Create the edges of the transition network:
        # Reduce the sig_test_set to have only N_input_nodes key parameter values:
        if len(pnet.input_node_inds) > N_input_nodes:
            ifoo = np.all((np.round(sig_test_set)[:, N_input_nodes:] == 0), axis=1).nonzero()[0]
            sig_test_set = sig_test_set[ifoo, :]

        dt = 5.0e-3
        dt_samp = 0.1
        delta_sig = 60.0
        t_relax = 20.0
        verbose = True
        match_tol = 0.2
        remove_inaccessible_states = False
        graph_layout = 'dot'

        save_graph_file = os.path.join(save_path, f'nfsm_{net_name}.gml')

        if verbose:
            print('Creating NFSM...')

        transition_edges_set, pert_edges_set, G_nx = smach.create_transition_network(states_dict,
                                                                                     sig_test_set,
                                                                                     solsM_all,
                                                                                     dt=dt,
                                                                                     delta_sig=delta_sig,
                                                                                     t_relax=t_relax,
                                                                                     dt_samp=dt_samp,
                                                                                     verbose=verbose,
                                                                                     match_tol=match_tol,
                                                                                     d_base=d_base,
                                                                                     n_base=n_base,
                                                                                     beta_base=beta_base,
                                                                                     remove_inaccessible_states=remove_inaccessible_states,
                                                                                     save_graph_file=save_graph_file,

                                                                                     )

        if smach._solsM_all.shape[1] != solsM_all.shape[1]:
            charM_xtra = ['undetermined' for i in range(smach._solsM_all.shape[1] - solsM_all.shape[1] + 1)]
            charM_all = np.hstack((charM_all, charM_xtra))

        nodes_list_fsm = list(G_nx.nodes())
        edges_list_fsm = list(G_nx.edges)
        mono_edge = False

        if mono_edge is True:
            mono_lab = 'monoedge'
        else:
            mono_lab = 'multiedge'

        # fimg = '.svg'
        fimg = f'.{img_fmt}'

        graph_layout = 'dot'
        # graph_layout = 'neato'

        # Next we'd like to analyze the NFSM to identify cycles and hierarchical level (flow tendancy):
        edges_fsm_di = sorted(nx.DiGraph(G_nx).edges)

        N_fsm_nodes = len(np.unique(edges_fsm_di))
        fnet = BasicNet(N_fsm_nodes)
        fnet.build_network_from_edges(edges_fsm_di)
        fnet.characterize_graph(count_cycles=True,
                                cycle_length_bound=10)

        if verbose:
            print("Hierarchical node level of NFSM:")
            print(fnet.hier_node_level)
            print("----")

        fsm_net_dict = {'N Cycles': fnet.N_cycles,
                        'N Nodes': fnet.N_nodes,
                        'N Edges': fnet.N_edges,
                        'Connectivity fraction': fnet.N_edges / fnet.N_nodes ** 2,
                        'Out-Degree Max': fnet.out_dmax,
                        'In-Degree Max': fnet.in_dmax,
                        'Democracy Coefficient': fnet.dem_coeff,
                        'Hierarchical Incoherence': fnet.hier_incoherence,
                        }

        pert_edges_di = []
        for ei, ej, _, _ in pert_edges_set:
            if (ei, ej) not in pert_edges_di:
                pert_edges_di.append((ei, ej))

        N_pfsm_nodes = len(np.unique(pert_edges_di))
        if N_pfsm_nodes != 0:
            pfnet = BasicNet(N_pfsm_nodes)
            pfnet.build_network_from_edges(pert_edges_di)
            pfnet.characterize_graph(count_cycles=True,
                                     cycle_length_bound=10)

            pfsm_net_dict = {'N Cycles': pfnet.N_cycles,
                             'N Nodes': pfnet.N_nodes,
                             'N Edges': pfnet.N_edges,
                             'Connectivity fraction': pfnet.N_edges / pfnet.N_nodes ** 2,
                             'Out-Degree Max': pfnet.out_dmax,
                             'In-Degree Max': pfnet.in_dmax,
                             'Democracy Coefficient': pfnet.dem_coeff,
                             'Hierarchical Incoherence': pfnet.hier_incoherence,
                             }

            save_perturbation_net_image = os.path.join(save_path, f'Pert_Net_{net_name}_' + mono_lab + fimg)
            G_pert = smach.plot_state_perturbation_network(pert_edges_set,
                                                           charM_all,
                                                           nodes_listo=nodes_list_fsm,
                                                           save_file=save_perturbation_net_image,
                                                           graph_layout=graph_layout,
                                                           mono_edge=mono_edge,
                                                           constraint=True,
                                                           concentrate=False,
                                                           rank='same',
                                                           # node_colors=pfnet.hier_node_level.tolist()
                                                           )

        else:
            pfsm_net_dict = {'N Cycles': None,
                             'N Nodes': None,
                             'N Edges': None,
                             'Connectivity fraction': None,
                             'Out-Degree Max': None,
                             'In-Degree Max': None,
                             'Democracy Coefficient': None,
                             'Hierarchical Incoherence': None,
                             }

        save_transition_net_image = os.path.join(save_path, f'Trans_Net_{net_name}_' + mono_lab + fimg)
        G_gv = smach.plot_state_transition_network(nodes_list_fsm,
                                                   edges_list_fsm,
                                                   charM_all,
                                                   save_file=save_transition_net_image,
                                                   graph_layout=graph_layout,
                                                   mono_edge=mono_edge,
                                                   constraint=True,
                                                   concentrate=False,
                                                   rank='same',
                                                   # node_colors=fnet.hier_node_level.tolist()
                                                   )

        save_microarray_image = os.path.join(save_path, f'Microarray_{net_name}_smach.{img_fmt}')
        fig, ax = pnet.plot_sols_array(smach._solsM_all,
                             gene_inds=pnet.noninput_node_inds,
                             figsave=save_microarray_image,
                             cmap=None,
                             save_format=img_fmt
                             )
        plt.close(fig)

        save_inputs_image = os.path.join(save_path, f'Inputs_{net_name}_smach.{img_fmt}')
        fig, ax = pnet.plot_pixel_matrix(sig_test_set.T,
                             pnet.input_node_inds,
                             figsave=save_inputs_image,
                             cmap=None
                             )
        plt.close(fig)

        num_sols = smach._solsM_all.shape[1]
        if N_pfsm_nodes != 0:
            num_pert_states = G_pert.number_of_nodes()
        else:
            num_pert_states = 0

        graph_data = {'Index': frame_i,
                      'Base File': net_name,
                      'Input probability': p_in_type.name,
                      'Output probability': p_out_type.name,
                      'N Cycles': pnet.N_cycles,
                      'N Nodes': pnet.N_nodes,
                      'N Edges': pnet.N_edges,
                      'Connectivity fraction': pnet.N_edges / pnet.N_nodes ** 2,
                      'Out-Degree Max': pnet.out_dmax,
                      'In-Degree Max': pnet.in_dmax,
                      'Democracy Coefficient': pnet.dem_coeff,
                      'Hierarchical Incoherence': pnet.hier_incoherence,
                      'N Unique Solutions': num_sols,
                      'N Pert States': num_pert_states}

        # Final thing we'd like to do is for each state, nip out nodes with low
        # expression to make it its own graph. Then perform network analysis on the network
        # arising from the state network:
        net_data = {'State': [-1],
                    'N Cycles': [pnet.N_cycles],
                    'N Nodes': [pnet.N_nodes],
                    'N Edges': [pnet.N_edges],
                    'Connectivity fraction': [pnet.N_edges / pnet.N_nodes ** 2],
                    'Out-Degree Max': [pnet.out_dmax],
                    'In-Degree Max': [pnet.in_dmax],
                    'Democracy Coefficient': [pnet.dem_coeff],
                    'Hierarchical Level': [-99],
                    'Hierarchical Incoherence': [pnet.hier_incoherence],
                    }

        for si, (state_i, hier_i) in enumerate(zip(smach._solsM_all.T, fnet.hier_node_level)):
            nodes_clip = (state_i[pnet.noninput_node_inds] < 0.1).nonzero()[0]

            print(f'State {si}, clip nodes {len(nodes_clip)}')

            if len(nodes_clip) < int(N_nodes/2):

                G_state = copy.deepcopy(pnet.GG)

                G_state.remove_nodes_from(np.asarray(pnet.nodes_list)[nodes_clip])

                new_edges = sorted(G_state.edges)

                N_sub_nodes = len(np.unique(new_edges))
                bnet = BasicNet(N_sub_nodes)
                bnet.build_network_from_edges(new_edges)
                bnet.characterize_graph(count_cycles=True,
                                        cycle_length_bound=10)

                figi, ax = bnet.plot_degree_distributions()
                deg_dist_fig_i = os.path.join(save_path, f'degree_dist{fname_base}_state_{si}.png')
                plt.savefig(deg_dist_fig_i, dpi=300, transparent=True, format='png')
                plt.close(figi)

                net_data['State'].append(si)
                net_data['N Cycles'].append(bnet.N_cycles)
                net_data['N Nodes'].append(bnet.N_nodes)
                net_data['N Edges'].append(bnet.N_edges)
                net_data['Connectivity fraction'].append(bnet.N_edges / bnet.N_nodes ** 2)
                net_data['Out-Degree Max'].append(bnet.out_dmax)
                net_data['In-Degree Max'].append(bnet.in_dmax)
                net_data['Democracy Coefficient'].append(bnet.dem_coeff)
                net_data['Hierarchical Level'].append(hier_i)
                net_data['Hierarchical Incoherence'].append(bnet.hier_incoherence)

        df = pd.DataFrame.from_dict(net_data)

        dframetest_file = os.path.join(save_path, f'subnetwork_data_{net_name}.csv')
        df.to_csv(dframetest_file)

        print(f"Completed workframe for {net_name}.")
        print('*****')
        print('')

        return graph_data, fsm_net_dict, pfsm_net_dict, net_data


    def work_frame(self,
                   pnet: ProbabilityNet,
                   save_path: str,
                   fname_base: str,
                   i_frame: int=0,
                   verbose: bool=True,
                   reduce_dims: bool = False,
                   beta_base: float | list = 0.25,
                   n_base: float | list = 15.0,
                   d_base: float | list = 1.0,
                   edge_types: list[EdgeType]|None = None,
                   edge_type_search: bool = True,
                   edge_type_search_iterations: int = 5,
                   find_solutions: bool = True,
                   knockout_experiments: bool = True,
                   sol_search_tol: float = 1.0e-15,
                   N_search_space: int = 2,
                   N_round_unique_sol: int = 1,
                   sol_unique_tol: float = 1.0e-1,
                   sol_ko_tol: float = 1.0e-1,
                   constraint_vals: list[float]|None = None,
                   constraint_inds: list[int]|None = None,
                   signal_constr_vals: list | None = None,
                   update_string: str|None = None,
                   node_type_dict: dict|None = None,
                   extra_verbose: bool=False,
                   coupling_type: CouplingType=CouplingType.mix1,
                   label_edges: bool = False,
                   search_cycle_nodes_only: bool = False,
                   cluster_threshhold: float = 0.1,
                   cluster_method: str = 'distance'
                   ):
        '''
        A single frame of the workflow
        '''

        if constraint_vals is not None and constraint_inds is not None:
            if len(constraint_vals) != len(constraint_inds):
                raise Exception("Node constraint values must be same length as constrained node indices!")

        if verbose is True:
            print(f'Iteration {i_frame}...')
            # print(update_string)

        # set node types to the network:
        pnet.set_node_types(node_type_dict=node_type_dict)

        if edge_types is None:
            if edge_type_search is False:
                # Create random edge types:
                edge_types = pnet.get_edge_types(p_acti=0.5)

            else:
                numsols, multisols = multistability_search(pnet,
                                                          N_multi=1,
                                                          sol_tol=sol_unique_tol,
                                                          N_iter=edge_type_search_iterations,
                                                          verbose=extra_verbose,
                                                          beta_base=beta_base,
                                                          n_base=n_base,
                                                          d_base=d_base,
                                                          N_space=N_search_space,
                                                          N_round_unique_sol=N_round_unique_sol,
                                                          search_tol=sol_search_tol,
                                                          constraint_vals=constraint_vals,
                                                          constraint_inds=constraint_inds,
                                                           signal_constr_vals=signal_constr_vals,
                                                          coupling_type=coupling_type,
                                                          )

                i_max = (np.asarray(numsols) == np.max(numsols)).nonzero()[0]

                _, edge_types = multisols[i_max[0]]

        # set edge types to the network:
        pnet.edge_types = edge_types
        pnet.set_edge_types(pnet.edge_types)

        # rebuild the model with the new edge_types:
        # Get the adjacency matrices for this model:
        A_add_s, A_mul_s, A_full_s = pnet.build_adjacency_from_edge_type_list(edge_types,
                                                                              pnet.edges_index,
                                                          coupling_type=coupling_type)
        # build the analytical model for this network:
        pnet.build_analytical_model(A_add_s, A_mul_s)

        # save the randomly generated network as a text file:
        gfile = f'network_{fname_base}.gml'
        save_gfile = os.path.join(save_path, gfile)
        pnet.save_network(save_gfile)

        # Save the network images:
        graph_net = f'hier_graph_{fname_base}.png'
        save_graph_net = os.path.join(save_path, graph_net)

        graph_net_c = f'circ_graph_{fname_base}.png'
        save_graph_net_circo = os.path.join(save_path, graph_net_c)

        # Highlight the hierarchical nature of the graph and info flow:
        gp=plot_network(pnet.nodes_list,
                        pnet.edges_list,
                        pnet.node_types,
                        pnet.edge_types,
                        node_vals = pnet.hier_node_level,
                        val_cmap = 'Greys_r',
                        save_path=save_graph_net,
                        layout='dot',
                        rev_font_color=True,
                        label_edges=label_edges
                        )

        # Highlight the existance of a "core" graph:
        cycle_tags = np.zeros(pnet.N_nodes)
        cycle_tags[pnet.nodes_in_cycles] = 1.0

        gp=plot_network(pnet.nodes_list,
                        pnet.edges_list,
                        pnet.node_types,
                        pnet.edge_types,
                        node_vals = cycle_tags,
                        val_cmap = 'Blues',
                        save_path=save_graph_net_circo,
                        layout='circo',
                        rev_font_color=False,
                        vminmax = (0.0, 1.0),
                        label_edges=label_edges
                        )

        # Plot and save the degree distribution for this graph:
        graph_deg = f'degseq_{fname_base}.png'
        save_graph_deg = os.path.join(save_path, graph_deg)
        fig, ax = pnet.plot_degree_distributions()
        fig.savefig(save_graph_deg, dpi=300, transparent=True, format='png')
        plt.close(fig)

        if find_solutions:

            if reduce_dims:  # If reduce dimensions then perform this calculation
                pnet.reduce_model_dimensions()

            if pnet._reduced_dims and pnet._solved_analytically is False:  # if dim reduction was attempted and was successful...
                # determine the size of the reduced dimensions vector:
                N_reduced_dims = len(pnet._dcdt_vect_reduced_s)

            elif pnet._solved_analytically:
                N_reduced_dims = pnet.N_nodes

            else:  # otherwise assign it to NaN
                N_reduced_dims = np.nan

            eqn_render = f'Eqn_{fname_base}.png'
            save_eqn_render = os.path.join(save_path, eqn_render)

            eqn_renderr = f'Eqnr_{fname_base}.png'
            save_eqn_renderr = os.path.join(save_path, eqn_renderr)

            eqn_net_file = f'Eqns_{fname_base}.csv'
            save_eqn_net = os.path.join(save_path, eqn_net_file)

            pnet.save_model_equations(save_eqn_render, save_eqn_renderr, save_eqn_net)

            soln_fn = f'soldat_{fname_base}.csv'
            save_solns = os.path.join(save_path, soln_fn)

            solsM, sol_M0_char, sol_0 = pnet.solve_probability_equms(constraint_inds=constraint_inds,
                                                                     constraint_vals=constraint_vals,
                                                                     signal_constr_vals=signal_constr_vals,
                                                                     d_base=d_base,
                                                                     n_base=n_base,
                                                                     beta_base=beta_base,
                                                                     N_space=N_search_space,
                                                                     search_tol=sol_search_tol,
                                                                     sol_tol=sol_unique_tol,
                                                                     N_round_sol=N_round_unique_sol,
                                                                     save_file=save_solns,
                                                                     verbose=extra_verbose,
                                                                     search_main_nodes_only=search_cycle_nodes_only
                                                                     )


            # cluster close solutions to avoid degeneracy in solutions:
            solsM = self.find_unique_sols(solsM,
                                          cluster_threshhold=cluster_threshhold,
                                          cluster_method=cluster_method,
                                          N_round_sol=N_round_unique_sol)

            if len(solsM):
                num_sols = solsM.shape[1]
            else:
                num_sols = 0

            fign = f'solArray_{fname_base}.png'
            figsave = os.path.join(save_path, fign)

            fig, ax = pnet.plot_sols_array(solsM, gene_inds=None, figsave=figsave)
            plt.close(fig)

            # Perform knockout experiments, if desired:
            if knockout_experiments:
                gko = GeneKnockout(pnet)
                knockout_sol_set, knockout_matrix, ko_header_o = gko.gene_knockout_ss_solve(
                                                                       Ns=N_search_space,
                                                                       tol=sol_search_tol,
                                                                       d_base=d_base,
                                                                       n_base=n_base,
                                                                       beta_base=beta_base,
                                                                       round_unique_sol=N_round_unique_sol,
                                                                       verbose=extra_verbose,
                                                                       sol_tol=sol_ko_tol,
                                                                       save_file_basename=None,
                                                                       constraint_vals=constraint_vals,
                                                                       constraint_inds=constraint_inds,
                                                                       signal_constr_vals=signal_constr_vals,
                                                                       cluster_threshhold = cluster_threshhold,
                                                                       cluster_method = cluster_method
                                                                       )

                ko_file = f'knockoutArrays{fname_base}.png'
                save_ko = os.path.join(save_path, ko_file)
                fig, ax = gko.plot_knockout_arrays(knockout_sol_set, figsave=save_ko)
                plt.close(fig)

                # save the knockout data to a file:
                ko_header = ''
                for si in ko_header_o:
                    ko_header += si
                dat_knockout_save = os.path.join(save_path, f'knockoutData_f{fname_base}.csv')
                np.savetxt(dat_knockout_save, knockout_matrix, delimiter=',', header=ko_header)

        else:
            num_sols = 0
            N_reduced_dims = 0

        graph_data = {'Index': i_frame,
                      'Base File': fname_base,
                      'Graph Type': pnet._graph_type.name,
                      'N Cycles': pnet.N_cycles,
                      'N Nodes': pnet.N_nodes,
                      'N Edges': pnet.N_edges,
                      'Out-Degree Max': pnet.out_dmax,
                      'In-Degree Max': pnet.in_dmax,
                      'Democracy Coefficient': pnet.dem_coeff,
                      'Hierarchical Incoherence': pnet.hier_incoherence,
                      'N Unique Solutions': num_sols,
                      'N Reduced Dims': N_reduced_dims}

        if verbose is True and update_string is not None:
            print(f'{update_string} Nsols: {num_sols}')

        return graph_data

    def write_network_data_file(self, dat_frame_list: list[dict], save_path: str):
        '''

        '''

        # networks_data_file = os.path.join(save_path, 'networks_data_file.csv')

        # Open a file in write mode.
        with open(save_path, 'w') as f:
            # Write all the dictionary keys in a file with commas separated.
            f.write(','.join(dat_frame_list[0].keys()))
            f.write('\n')  # Add a new line
            for row in dat_frame_list:
                # Write the values in a row.
                f.write(','.join(str(x) for x in row.values()))
                f.write('\n')  # Add a new line

    def get_all_graph_files(self, read_path: str):
        '''
        Returns a list of all graph file names in a directory. These can be used to re-load
        graphs for further analysis.
        '''

        # list to store files
        graph_files_list = []
        # Iterate directory
        for file in os.listdir(read_path):
            # check only text files
            if file.endswith('.gml'):
                graph_files_list.append(file)

        return graph_files_list

    def find_unique_sols(self,
                         solsM,
                         cluster_threshhold: float=0.1,
                         cluster_method: str='distance',
                         N_round_sol: int=2):
        '''

        '''

        if solsM.shape[1] > 1:
            print("resizing by clustering")
            unique_sol_clusters = fclusterdata(solsM.T, t=cluster_threshhold, criterion=cluster_method)

            cluster_index = np.unique(unique_sol_clusters)

            cluster_pool = [[] for i in cluster_index]
            for i, clst_i in enumerate(unique_sol_clusters):
                cluster_pool[int(clst_i) - 1].append(i)

            solsM_all_unique = np.zeros((solsM.shape[0], len(cluster_pool)))

            for ii, sol_i in enumerate(cluster_pool):
                if len(sol_i):
                    solsM_all_unique[:, ii] = (np.mean(solsM[:, sol_i], 1))

            # redefine the solsM data structures:
            solsM = solsM_all_unique

            # # # first use numpy unique on rounded set of solutions to exclude similar cases:
            _, inds_solsM_all_unique = np.unique(np.round(solsM, N_round_sol), return_index=True, axis=1)
            solsM = solsM[:, inds_solsM_all_unique]
        else:
            print("not resizing by clustering")

        return solsM





