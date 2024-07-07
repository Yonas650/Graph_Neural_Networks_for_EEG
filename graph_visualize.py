import os
import networkx as nx
import matplotlib.pyplot as plt

#visualize a graph
def visualize_graph(graph_path, save_path=None):
    G = nx.read_graphml(graph_path)
    
    #extract node features for color mapping
    node_features = nx.get_node_attributes(G, 'feature')
    node_colors = [node_features[node] for node in G.nodes]
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  #position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.viridis, node_size=500, edge_color='grey')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to {save_path}")
    else:
        plt.show()

#load and visualize graph files
def visualize_all_graphs(graph_dir, save_dir=None):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for graph_file in os.listdir(graph_dir):
        if graph_file.endswith('.graphml'):
            graph_path = os.path.join(graph_dir, graph_file)
            print(f"Visualizing {graph_path}")
            
            if save_dir:
                save_path = os.path.join(save_dir, f"{os.path.splitext(graph_file)[0]}.png")
                visualize_graph(graph_path, save_path)
            else:
                visualize_graph(graph_path)

graph_dir = "graph_files"  
save_dir = "graph_images" 

visualize_all_graphs(graph_dir, save_dir)
