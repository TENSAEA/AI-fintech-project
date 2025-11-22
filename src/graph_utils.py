import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def build_transaction_graph(df):
    """
    Builds a NetworkX graph from transaction data.
    """
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        G.add_edge(row['sender_id'], row['receiver_id'], 
                   amount=row['amount'], 
                   timestamp=row['timestamp'],
                   is_fraud=row['is_fraud'])
    return G

def detect_rings(G, min_cycle_length=2):
    """
    Detects simple cycles (rings) in the graph.
    """
    try:
        cycles = list(nx.simple_cycles(G))
        # Filter by length if needed
        rings = [c for c in cycles if len(c) >= min_cycle_length]
        return rings
    except Exception as e:
        print(f"Error detecting rings: {e}")
        return []

def visualize_graph_plotly(G, highlight_nodes=None):
    """
    Visualizes the graph using Plotly.
    """
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        if highlight_nodes and node in highlight_nodes:
            node_color.append('red')
        else:
            node_color.append('blue')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2))
            
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig
