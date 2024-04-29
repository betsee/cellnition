#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines a top-level handler that can perform various workflows pertaining to network generation,
analytis, model generation, solution finding, searching, knockout experiments, and other functions.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from cellnition.science.network_models.network_enums import (EdgeType,
                                                             GraphType,
                                                             NodeType,
                                                             InterFuncType,
                                                             CouplingType)
from cellnition.science.network_models.probability_networks import ProbabilityNet
from cellnition.science.networks_toolbox.netplot import plot_network
from cellnition.science.networks_toolbox.gene_knockout import GeneKnockout
from cellnition.science.networks_toolbox.phase_space_searches import multistability_search

# FIXME: document throughout

class NetworkWorkflow(object):
    '''

    '''
    def __init__(self, save_path: str):
        '''

        '''
        self._save_path = save_path

    def scalefree_graph_gen(self,
                            N_nodes: int,
                            b_param: float,
                            g_param: float,
                            delta_in: float,
                            delta_out: float,
                            i: int,
                            interaction_function_type: InterFuncType = InterFuncType.logistic,
                            coupling_type: CouplingType = CouplingType.mixed):
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
                           coupling_type: CouplingType = CouplingType.mixed
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
                                  coupling_type: CouplingType=CouplingType.mixed,
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
                             coupling_type: CouplingType = CouplingType.mixed,
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
                   coupling_type: CouplingType=CouplingType.mixed,
                   label_edges: bool = False,
                   search_cycle_nodes_only: bool = False
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
                                                                  search_cycle_nodes_only=search_cycle_nodes_only
                                                                  )


            if len(solsM):
                num_sols = solsM.shape[1]
            else:
                num_sols = 0

            fign = f'solArray_{fname_base}.png'
            figsave = os.path.join(save_path, fign)

            fig, ax = pnet.plot_sols_array(solsM, figsave)
            plt.close(fig)

            # Perform knockout experiments, if desired:
            if knockout_experiments:
                gko = GeneKnockout(pnet)
                knockout_sol_set, knockout_matrix = gko.gene_knockout_ss_solve(
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
                                                                       signal_constr_vals=signal_constr_vals
                                                                       )

                ko_file = f'knockoutArrays{fname_base}.png'
                save_ko = os.path.join(save_path, ko_file)
                fig, ax = gko.plot_knockout_arrays(knockout_sol_set, figsave=save_ko)
                plt.close(fig)

                # save the knockout data to a file:
                dat_knockout_save = os.path.join(save_path, f'knockoutData_f{fname_base}.csv')
                np.savetxt(dat_knockout_save, knockout_matrix, delimiter=',')

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





