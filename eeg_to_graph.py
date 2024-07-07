import numpy as np
import networkx as nx
from scipy.stats import entropy
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

#create a graph from EEG
def create_graph(data):
    G = nx.Graph()
    num_channels = data.shape[0]  #number of EEG channels(electrodes)
    print("Number of electrodes", num_channels)
    for i in range(num_channels):
        channel_entropy = entropy(data[i])
        channel_entropy = np.nan_to_num(channel_entropy, nan=0.0, posinf=0.0, neginf=0.0)
        G.add_node(i, feature=float(channel_entropy))  # Convert to float
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            correlation = np.corrcoef(data[i], data[j])[0, 1]
            if correlation > 0.5:  #threshold for adding edges
                G.add_edge(i, j, weight=float(correlation))  #convert to float
    return G

#load and convert data for each subject
def convert_eeg_to_graphs(data_path, graph_path, n_sub):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    for sub_id in range(n_sub):
        X_path = os.path.join(data_path, f'X_PS_SR_{sub_id}.npy')
        y_path = os.path.join(data_path, f'y_PS_{sub_id}.npy')
        X_train = np.load(X_path).squeeze(-1)
        y_train = np.load(y_path)

        print(f"Subject {sub_id} Data Shape: {X_train.shape}")
        print(f"Subject {sub_id} Labels Shape: {y_train.shape}")

        if X_train.size == 0 or y_train.size == 0:
            print(f"Skipping subject {sub_id} due to empty data or labels")
            continue

        for trial in range(min(X_train.shape[0], y_train.shape[0])):
            trial_data = X_train[trial]  #EEG data for a single trial
            trial_label = int(y_train[trial])  #ensuring label is an integer
            graph = create_graph(trial_data)
            #save the graph in GraphML format
            graph_file_path = os.path.join(graph_path, f'graph_sub_{sub_id}_trial_{trial}.graphml')
            nx.write_graphml(graph, graph_file_path)
            print(f"Saved graph for subject {sub_id}, trial {trial}")

data_path = "np_files"  
graph_path = "graph_files"  
n_sub = 28 

convert_eeg_to_graphs(data_path, graph_path, n_sub)
