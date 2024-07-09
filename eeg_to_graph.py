import numpy as np
import networkx as nx
from scipy.stats import entropy
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_plv(signal1, signal2):
    phase1 = np.angle(signal1)
    phase2 = np.angle(signal2)
    plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
    return plv

def create_graph_with_plv(data):
    G = nx.Graph()
    num_channels = data.shape[0]  #number of EEG channels (electrodes)
    print("Number of electrodes", num_channels)
    node_entropies = []
    for i in range(num_channels):
        channel_entropy = entropy(data[i])
        channel_entropy = np.nan_to_num(channel_entropy, nan=0.0, posinf=0.0, neginf=0.0)
        G.add_node(i, feature=float(channel_entropy))  #convert to float
        node_entropies.append(channel_entropy)
    correlation_coeffs = []
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            plv = calculate_plv(data[i], data[j])
            correlation_coeffs.append(plv)
            if plv > 0.5:  #threshold for adding edges
                G.add_edge(i, j, weight=float(plv))  #convert to float
    return G, correlation_coeffs, node_entropies

def average_eeg_data(X_data, y_data, condition):
    condition_indices = np.where(y_data == condition)[0]
    condition_data = X_data[condition_indices]
    averaged_data = np.mean(condition_data, axis=0)
    return averaged_data

def convert_eeg_to_graphs(data_path, graph_path, image_path, n_sub):
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    all_X_data = []
    all_y_data = []
    all_correlation_coeffs = []
    all_entropies = []

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

        all_X_data.append(X_train)
        all_y_data.append(y_train)

    all_X_data = np.concatenate(all_X_data, axis=0)
    all_y_data = np.concatenate(all_y_data, axis=0)

    condition_correlation_coeffs = {condition: [] for condition in range(4)}
    condition_entropies = {condition: [] for condition in range(4)}

    for condition in range(4):
        averaged_data = average_eeg_data(all_X_data, all_y_data, condition)
        graph, correlation_coeffs, node_entropies = create_graph_with_plv(averaged_data)
        all_correlation_coeffs.extend(correlation_coeffs)
        condition_correlation_coeffs[condition].extend(correlation_coeffs)
        all_entropies.extend(node_entropies)
        condition_entropies[condition].extend(node_entropies)
        graph_file_path = os.path.join(graph_path, f'graph_condition_{condition}.graphml')
        nx.write_graphml(graph, graph_file_path)
        print(f"Saved graph for condition {condition}")

        #visualize and save the graph image
        node_features = nx.get_node_attributes(graph, 'feature')
        node_colors = [node_features[node] for node in graph.nodes]

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.viridis, node_size=500, edge_color='grey')
        image_file_path = os.path.join(image_path, f'graph_condition_{condition}.png')
        plt.savefig(image_file_path)
        plt.close()
        print(f"Saved graph image for condition {condition}")

    #plot and save the histogram for the correlation coefficients
    plt.figure(figsize=(10, 6))
    plt.hist(all_correlation_coeffs, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    histogram_path = os.path.join(image_path, 'correlation_histogram.png')
    plt.savefig(histogram_path)
    plt.close()
    print(f"Saved histogram of correlation coefficients")

    #plot and save histograms for each condition
    for condition in range(4):
        plt.figure(figsize=(10, 6))
        plt.hist(condition_correlation_coeffs[condition], bins=50, color='blue', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of Correlation Coefficients - Condition {condition}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        histogram_path = os.path.join(image_path, f'correlation_histogram_condition_{condition}.png')
        plt.savefig(histogram_path)
        plt.close()
        print(f"Saved histogram of correlation coefficients for condition {condition}")

    #plot and save the histogram for the entropy
    plt.figure(figsize=(10, 6))
    plt.hist(all_entropies, bins=50, color='green', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Entropies')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    entropy_histogram_path = os.path.join(image_path, 'entropy_histogram.png')
    plt.savefig(entropy_histogram_path)
    plt.close()
    print(f"Saved histogram of entropies")

    #plot and save histograms for entropy for each condition
    for condition in range(4):
        plt.figure(figsize=(10, 6))
        plt.hist(condition_entropies[condition], bins=50, color='green', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of Entropies - Condition {condition}')
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        entropy_histogram_path = os.path.join(image_path, f'entropy_histogram_condition_{condition}.png')
        plt.savefig(entropy_histogram_path)
        plt.close()
        print(f"Saved histogram of entropies for condition {condition}")

data_path = "np_files"
graph_path = "graph_files"
image_path = "graph_images"
n_sub = 28

convert_eeg_to_graphs(data_path, graph_path, image_path, n_sub)
