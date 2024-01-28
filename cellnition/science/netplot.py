import pygraphviz as pgv
import numpy as np
from numpy import ndarray
from matplotlib import colormaps
from matplotlib import colors
from cellnition.science.network_enums import EdgeType, GraphType, NodeType

def plot_network(nodes_list: list|ndarray,
                 edge_list: list|ndarray,
                 nodes_type: list|ndarray,
                 edges_type: list|ndarray,
                 node_vals: list|ndarray|None = None,
                 val_cmap: str|None = None,
                 path_plot_edges: list|ndarray|None = None,
                 dpi: int|float=300,
                 save_path: str|None=None,
                 layout: str='dot',
                 vminmax: tuple|None = None,
                 rev_font_color: bool=False):
    '''

    layout options:
    'dot'
    "fdp"
    'neato'
    '''

    G = pgv.AGraph(strict=False,
                   splines=True,
                   directed=True,
                   concentrate=False,
                   # nodesep=0.1,
                   # ranksep=0.3,
                   dpi=dpi)

    net_font_name = 'DejaVu Sans'
    node_font_size = 16
    tit_font_size = 24
    net_layout = 'TB'
    edge_width = 2.0

    if node_vals is not None:
        if vminmax is None:
            vmin = np.min(node_vals)
            vmax = np.max(node_vals)
        else:
            vmin = vminmax[0]
            vmax = vminmax[1]

        if val_cmap is None:
            cmap = colormaps['Greys'] # default colormap
        else:
            cmap = colormaps[val_cmap]

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    else:
        cmap = None
        norm = None

    node_dict_gene = {
    'node_font_color': 'Black',
    'node_color': 'GhostWhite',
    'node_shape': 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_signal = {
    'node_font_color': 'Black',
    'node_color': 'LemonChiffon',
    'node_shape': 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_cycle = {
    'node_font_color': 'Black',
    'node_color': 'PaleTurquoise',
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
    'node_color' : 'Turquoise',
    'node_shape' : 'ellipse',
    'outline_color': 'Black'
    }

    node_dict_process = {
    'node_font_color' : 'White',
    'node_color' : 'DarkSlateGrey',
    'node_shape' : 'rect',
    'outline_color': 'None'
    }

    node_dict_factor = {
    'node_font_color' : 'Black',
    'node_color' : 'GhostWhite',
    'node_shape' : 'diamond',
    'outline_color': 'Black'
    }

    node_plot_dict = {NodeType.gene.value: node_dict_gene,
                      NodeType.signal.value: node_dict_signal,
                      NodeType.sensor.value: node_dict_sensor,
                      NodeType.process.value: node_dict_process,
                      NodeType.effector.value: node_dict_effector,
                      NodeType.root.value: node_dict_root,
                      NodeType.cycle.value: node_dict_cycle,
                      NodeType.factor.value: node_dict_factor}

    for ni, (nn, nt) in enumerate(zip(nodes_list, nodes_type)):

        nde_dict = node_plot_dict[nt.value]

        if node_vals is None:
            nde_color = nde_dict['node_color']
            nde_outline = nde_dict['outline_color']
            nde_font_color = nde_dict['node_font_color']

        else:
            nde_color = colors.rgb2hex(cmap(norm(node_vals[ni])))
            nde_outline = 'Black'

            if rev_font_color is False:
                if norm(node_vals[ni]) <= 0.6:
                    nde_font_color = 'Black'
                else:
                    nde_font_color = 'White'

            else:
                if norm(node_vals[ni]) >= 0.6:
                    nde_font_color = 'Black'
                else:
                    nde_font_color = 'White'

        G.add_node(nn,
                   style='filled',
                   fillcolor=nde_color,
                   color=nde_outline,
                   shape=nde_dict['node_shape'],
                   fontcolor=nde_font_color,
                   fontname=net_font_name,
                   fontsize=node_font_size,
                   )

    for (ei, ej), et in zip(edge_list, edges_type):

        if et is EdgeType.A or et is EdgeType.As:
            G.add_edge(ei, ej, arrowhead='dot', color='blue', penwidth=edge_width)

        elif et is EdgeType.I or et is EdgeType.Is:
            G.add_edge(ei, ej, arrowhead='tee', color='red', penwidth=edge_width)

        elif et is EdgeType.N:
            G.add_edge(ei, ej, arrowhead='normal', color='black', penwidth=edge_width)

        else:
            raise Exception('Edge type not found.')

    G.layout(prog=layout) # default to neato

    if save_path is not None:
        G.draw(save_path)

    return G