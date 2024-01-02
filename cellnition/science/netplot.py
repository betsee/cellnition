import pygraphviz as pgv
from numpy import ndarray
from cellnition.science.enumerations import EdgeType, GraphType, NodeType

def plot_network(nodes_list: list|ndarray,
                 edge_list: list|ndarray,
                 nodes_type: list|ndarray,
                 edges_type: list|ndarray,
                 path_plot_edges: list|ndarray|None = None,
                 dpi: int|float=300,
                 save_path: str|None=None):
    '''

    '''

    G = pgv.AGraph(strict=False,
                   # splines=True,
                   directed=True,
                   concentrate=False,
                   nodesep=0.1,
                   ranksep=0.3,
                   dpi=dpi)

    net_font_name = 'DejaVu Sans'
    node_font_size = 16
    tit_font_size = 24
    net_layout = 'TB'
    edge_width = 2.0

    # graph_type = 'digraph'
    # concentrate = False
    # nodesep = 0.1
    # ranksep = 0.3
    # splines = True
    # strict = False
    # rankdir = net_layout

    node_dict_gene = {
    'node_font_color': 'Black',
    'node_color': 'GhostWhite',
    'node_shape': 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_sensor = {
    'node_font_color' : 'Black',
    'node_color' : 'LemonChiffon',
    'node_shape' : 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_root = {
    'node_font_color' : 'Black',
    'node_color' : 'Pink',
    'node_shape' : 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_effector = {
    'node_font_color' : 'Black',
    'node_color' : 'PaleTurquoise',
    'node_shape' : 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_process = {
    'node_font_color' : 'White',
    'node_color' : 'DarkSlateGrey',
    'node_shape' : 'rect',
    'outline_color': 'None'
    }

    node_plot_dict = {NodeType.gene.value: node_dict_gene,
                      NodeType.sensor.value: node_dict_sensor,
                      NodeType.process.value: node_dict_process,
                      NodeType.effector.value: node_dict_effector,
                      NodeType.root.value: node_dict_root}

    for ni, nt in zip(nodes_list, nodes_type):

        nde_dict = node_plot_dict[nt.value]

        G.add_node(ni,
                   style='filled',
                   fillcolor=nde_dict['node_color'],
                   color=nde_dict['outline_color'],
                   shape=nde_dict['node_shape'],
                   fontcolor=nde_dict['node_font_color'],
                   fontname=net_font_name,
                   fontsize=node_font_size,
                   )

    for (ei, ej), et in zip(edge_list, edges_type):

        if et is EdgeType.A:
            G.add_edge(ei, ej, arrowhead='dot', color='blue', penwidth=edge_width)

        elif et is EdgeType.I:
            G.add_edge(ei, ej, arrowhead='tee', color='red', penwidth=edge_width)

        elif et is EdgeType.N:
            G.add_edge(ei, ej, arrowhead='normal', color='black', penwidth=edge_width)

        else:
            raise Exception('Edge type not found.')

    G.layout() # default to neato

    if save_path is not None:
        G.draw(save_path)

    return G