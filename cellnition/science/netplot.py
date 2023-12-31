import pygraphviz as pgv

def plot_network(nodes_list, edge_list):
    '''

    '''

    # A = nx.nx_agraph.to_agraph(G)
    G = pgv.AGraph(strict=False, directed=True)

    net_font_name = 'DejaVu Sans'
    node_font_size = 16
    tit_font_size = 24
    net_layout = 'TB'
    edge_width = 2.0

    graph_type = 'digraph'
    concentrate = False
    nodesep = 0.1
    ranksep = 0.3
    splines = True
    strict = False
    rankdir = net_layout

    conc_node_font_color = 'Black'
    conc_node_color = 'LightCyan'
    conc_shape = 'ellipse'

    react_node_font_color = 'White'
    react_node_color = 'DarkSeaGreen'
    reaction_shape = 'rect'

    transp_node_font_color = 'White'
    transp_node_color = 'DarkSeaGreen'
    transporter_shape = 'diamond'

    chan_node_font_color = 'White'
    chan_node_color = 'DarkSeaGreen'
    channel_shape = 'pentagon'

    ed_node_font_color = 'White'
    ed_node_color = 'DarkSeaGreen'
    ed_shape = 'hexagon'

    vmem_node_font_color = 'White'
    vmem_node_color = 'maroon'
    vmem_shape = 'oval'

    G.add_node(ni,
               style='filled',
               color=react_node_color,
               shape=reaction_shape,
               fontcolor=react_node_font_color,
               fontname=net_font_name,
               fontsize=node_font_size)

    G.add_edge(ei, ej, arrowhead='dot', color='blue', penwidth=edge_width)

    G.add_edge(ei, ej, arrowhead='tee', color='red', penwidth=edge_width)

    G.add_edge(ei, ej, arrowhead='normal', color='black', penwidth=edge_width)

    G.layout() # default to neato

    return G