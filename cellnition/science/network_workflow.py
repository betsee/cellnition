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
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
from cellnition.science.gene_networks import GeneNetworkModel
from cellnition.science.netplot import plot_network
from cellnition.science.gene_knockout import GeneKnockout
from cellnition.science.phase_space_searches import multistability_search

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
                            bi: float,
                            gi: float,
                            delta_in: float,
                            delta_out: float,
                            i: int):
        '''

        '''
        ai = 1.0 - bi - gi

        gmod = GeneNetworkModel(N_nodes,
                                graph_type=GraphType.scale_free,
                                edges=None,
                                beta=bi,
                                gamma=gi,
                                delta_in=delta_in,
                                delta_out=delta_out)


        dem_coeff = np.round(gmod.dem_coeff, 1)
        incoh = np.round(gmod.hier_incoherence, 1)
        fname_base = f'{i}_sf{N_nodes}_b{bi}_g{gi}_Ncycles{gmod.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: params {np.round(ai,2), bi, gi, delta_in, delta_out}, '
                         f'cycles: {gmod.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoh.: {incoh}')

        return gmod, update_string, fname_base

    def binomial_graph_gen(self,
                           N_nodes: int,
                           p_edge: float,
                           i: int):
        '''

        '''
        gmod = GeneNetworkModel(N_nodes, graph_type=GraphType.random, edges=None, p_edge=p_edge)

        dem_coeff = np.round(gmod.dem_coeff, 1)
        incoh = np.round(gmod.hier_incoherence, 1)
        fname_base = f'{i}_bino{N_nodes}_Ncycles{gmod.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: params {p_edge}, '
                         f'cycles: {gmod.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoherence: {incoh}')

        return gmod, update_string, fname_base

    def read_graph_from_file(self,
                             filename: str,
                             add_interactions: bool = True,
                             build_analytical: bool=False,
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
        gmod = GeneNetworkModel(N_nodes, edges=edges_list)

        # Assign node types to the network model:
        gmod.set_node_types(node_type_dict=node_type_dict)

        if build_analytical:
            # Build an analytical model using the edge_type and node_type assignments:
            gmod.build_analytical_model(edge_types=edge_types,
                                add_interactions=add_interactions,
                                node_type_dict=None
                               )

        else: # just set the edge types for the model:
            gmod.set_edge_types(edge_types, add_interactions=add_interactions)

        dem_coeff = np.round(gmod.dem_coeff, 1)
        incoh = np.round(gmod.hier_incoherence, 1)

        fname_base = f'{i}_bino{N_nodes}_Ncycles{gmod.N_cycles}_dem{dem_coeff}_incoh{incoh}'

        update_string = (f'{i}: cycles: {gmod.N_cycles}, '
                         f'dem_coeff: {dem_coeff}, '
                         f'incoherence: {incoh}')

        return gmod, update_string, fname_base

    def work_frame(self,
                   gmod: GeneNetworkModel,
                   save_path: str,
                   fname_base: str,
                   i_frame: int=0,
                   verbose: bool=True,
                   reduce_dims: bool = False,
                   Bi: float | list = 2.0,
                   ni: float|list = 3.0,
                   di: float|list = 1.0,
                   coi: float|list = 0.0,
                   ki: float|list = 10.0,
                   add_interactions: bool = True,
                   edge_types: list[EdgeType]|None = None,
                   edge_type_search: bool = True,
                   N_edge_search: int = 5,
                   find_solutions: bool = True,
                   knockout_experiments: bool = True,
                   sol_search_tol: float = 1.0e-15,
                   N_search_space: int = 3,
                   N_round_sol: int = 6,
                   N_round_unique_sol: int = 1,
                   sol_unique_tol: float = 1.0e-1,
                   sol_ko_tol: float = 1.0e-1,
                   constraint_vals: list[float]|None = None,
                   constraint_inds: list[int]|None = None,
                   update_string: str|None = None,
                   pure_gene_edges_only: bool = False,
                   node_type_dict: dict|None = None,
                   solver_method: str = 'Root',
                   extra_verbose: bool=False
                   ):
        '''
        A single frame of the workflow
        '''

        # if graph_type is GraphType.scale_free:
        #     # generate a graph for this frame
        #     gmod, update_string, fname_base = self.scalefree_graph_gen(N_nodes, bi, gi, delta_in, delta_out, i_frame)
        #
        # else: # generate a random (a.k.a binomial) graph
        #     gmod, update_string, fname_base = self.binomial_graph_gen(N_nodes, p_edge, i_frame)

        if constraint_vals is not None and constraint_inds is not None:
            if len(constraint_vals) != len(constraint_inds):
                raise Exception("Node constraint values must be same length as constrained node indices!")

        if verbose is True:
            print(f'Iteration {i_frame}...')
            # print(update_string)

        # set node types to the network:
        gmod.set_node_types(node_type_dict=node_type_dict, pure_gene_edges_only=pure_gene_edges_only)

        if edge_types is None:
            if edge_type_search is False:
                # Create random edge types:
                edge_types = gmod.get_edge_types(p_acti=0.5)

            else:
                gmod.create_parameter_vects(Bi=Bi, ni=ni, di=di, co=coi, ki=ki)
                numsols, multisols = multistability_search(gmod, 1,
                                                                tol=sol_unique_tol,
                                                                N_iter=N_edge_search,
                                                                verbose=extra_verbose,
                                                                add_interactions=add_interactions,
                                                                N_round_unique_sol=N_round_unique_sol,
                                                                unique_sols=True,
                                                                constraint_vals=constraint_vals,
                                                                constraint_inds=constraint_inds,
                                                                node_type_dict=node_type_dict
                                                           )

                i_max = (np.asarray(numsols) == np.max(numsols)).nonzero()[0]

                _, edge_types = multisols[i_max[0]]

        # set edge types to the network:
        gmod.edge_types = edge_types
        gmod.set_edge_types(gmod.edge_types, add_interactions)

        # save the randomly generated network as a text file:
        gfile = f'network_{fname_base}.gml'
        save_gfile = os.path.join(save_path, gfile)
        gmod.save_network(save_gfile)

        # Save the network images:
        graph_net = f'hier_graph_{fname_base}.png'
        save_graph_net = os.path.join(save_path, graph_net)

        graph_net_c = f'circ_graph_{fname_base}.png'
        save_graph_net_circo = os.path.join(save_path, graph_net_c)

        # Highlight the hierarchical nature of the graph and info flow:
        gp=plot_network(gmod.nodes_list,
                    gmod.edges_list,
                    gmod.node_types,
                    gmod.edge_types,
                    node_vals = gmod.hier_node_level,
                    val_cmap = 'Greys_r',
                    save_path=save_graph_net,
                    layout='dot',
                    rev_font_color=True
                   )

        # Highlight the existance of a "core" graph:
        cycle_tags = np.zeros(gmod.N_nodes)
        cycle_tags[gmod.nodes_in_cycles] = 1.0

        gp=plot_network(gmod.nodes_list,
            gmod.edges_list,
            gmod.node_types,
            gmod.edge_types,
            node_vals = cycle_tags,
            val_cmap = 'Blues',
            save_path=save_graph_net_circo,
            layout='circo',
            rev_font_color=False
           )

        # Plot and save the degree distribution for this graph:
        graph_deg = f'degseq_{fname_base}.png'
        save_graph_deg = os.path.join(save_path, graph_deg)
        fig, ax = gmod.plot_degree_distributions()
        fig.savefig(save_graph_deg, dpi=300, transparent=True, format='png')
        plt.close(fig)

        if find_solutions:

            gmod.build_analytical_model(edge_types=edge_types,
                                        add_interactions=add_interactions,
                                        node_type_dict=node_type_dict,
                                        pure_gene_edges_only=pure_gene_edges_only,
                                        )

            gmod.create_parameter_vects(Bi=Bi, ni=ni, di=di, co=coi, ki=ki)

            if reduce_dims:  # If reduce dimensions then perform this calculation
                gmod.reduce_model_dimensions()

            if gmod._reduced_dims and gmod._solved_analytically is False:  # if dim reduction was attempted and was successful...
                # determine the size of the reduced dimensions vector:
                N_reduced_dims = len(gmod.dcdt_vect_reduced_s)

            elif gmod._solved_analytically:
                N_reduced_dims = gmod.N_nodes

            else:  # otherwise assign it to zero
                N_reduced_dims = 0

            eqn_render = f'Eqn_{fname_base}.png'
            save_eqn_render = os.path.join(save_path, eqn_render)

            eqn_renderr = f'Eqnr_{fname_base}.png'
            save_eqn_renderr = os.path.join(save_path, eqn_renderr)

            eqn_net_file = f'Eqns_{fname_base}.csv'
            save_eqn_net = os.path.join(save_path, eqn_net_file)

            gmod.save_model_equations(save_eqn_render, save_eqn_renderr, save_eqn_net)

            if add_interactions is True:
                cmax = 1.5*np.max(gmod.in_degree_sequence)
            else:
                cmax = 1.5

            if constraint_inds is None or constraint_vals is None:
                print("Solving experiments *without* constraints")
                sols_0 = gmod.optimized_phase_space_search(Ns=N_search_space,
                                                   cmax=cmax,
                                                   round_sol=N_round_sol,
                                                   tol=sol_search_tol,
                                                   method=solver_method
                                                  )

            else: # otherwise, if node constraints are defined:
                print("Solving experiments *with* constraints")
                sols_0 = gmod.constrained_phase_space_search(constraint_vals,
                                                                   constraint_inds,
                                                                   Ns=N_search_space,
                                                                   cmax=cmax,
                                                                   tol=sol_search_tol,
                                                                   round_sol=N_round_sol,
                                                                   method=solver_method
                                                                   )

            soln_fn = f'soldat_{fname_base}.csv'
            save_solns = os.path.join(save_path, soln_fn)
            solsM = gmod.find_attractor_sols(sols_0,
                                             tol=sol_unique_tol,
                                             N_round=N_round_unique_sol,
                                             verbose=extra_verbose,
                                             unique_sols=True,
                                             save_file=save_solns)

            if len(solsM):
                num_sols = solsM.shape[1]
            else:
                num_sols = 0

            fign = f'solArray_{fname_base}.png'
            figsave = os.path.join(save_path, fign)

            fig, ax = gmod.plot_sols_array(solsM, figsave)
            plt.close(fig)

            # Perform knockout experiments:
            if add_interactions is True:
                cmax = 1.5 * np.max(gmod.in_degree_sequence)
            else:
                cmax = 1.5*(1 / np.max(np.asarray(gmod.d_vect)))

            if knockout_experiments:
                gko = GeneKnockout(gmod)
                knockout_sol_set, knockout_matrix = gko.gene_knockout_ss_solve(
                                                                           Ns=N_search_space,
                                                                       cmin=0.0,
                                                                       cmax=cmax,
                                                                       tol=sol_search_tol,
                                                                       round_sol=N_round_sol,
                                                                       round_unique_sol=N_round_unique_sol,
                                                                       verbose=extra_verbose,
                                                                       unique_sols=True,
                                                                       sol_tol=sol_ko_tol,
                                                                       save_file_basename=None,
                                                                       constraint_vals=constraint_vals,
                                                                       constraint_inds=constraint_inds,
                                                                       solver_method=solver_method
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
                      'Graph Type': gmod._graph_type.name,
                      'Alpha': np.round(1.0 - gmod._beta - gmod._gamma,2),
                      'Beta': gmod._beta, # this only applies for scale-free graphs
                      'Gamma': gmod._gamma,
                      'Delta in': gmod._delta_in,
                      'Delta out': gmod._delta_out,
                      'Edge p': gmod._p_edge, # this only applies for binomial graphs
                      'N Cycles': gmod.N_cycles,
                      'N Nodes': gmod.N_nodes,
                      'N Edges': gmod.N_edges,
                      'Out-Degree Max': gmod.out_dmax,
                      'In-Degree Max': gmod.in_dmax,
                      'Democracy Coefficient': gmod.dem_coeff,
                      'Hierarchical Incoherence': gmod.hier_incoherence,
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



    # gmod_list = []
    # for i, graph_file in enumerate(graph_files_list):
    #     print(i)
    #     read_gfile = os.path.join(read_path, graph_file)
    #     gmod = sim.read_network(read_gfile, add_interactions = True, build_analytical=True)

    #     # What we want to see is: can the model be reduced?
    #     gmod.reduce_model_dimensions()

    #     eqn_render = f'f{graph_file[0:-4]}_Eqn.png'
    #     save_eqn_render = os.path.join(save_path, eqn_render)

    #     eqn_renderr = f'f{graph_file[0:-4]}_Eqnr.png'
    #     save_eqn_renderr = os.path.join(save_path, eqn_renderr)

    #     eqn_net_file = f'{graph_file[0:-4]}_Eqns.csv'
    #     save_eqn_net = os.path.join(save_path, eqn_net_file)

    #     gmod.save_model_equations(save_eqn_render, save_eqn_renderr, save_eqn_net)

    #     gmod_list.append(gmod)




