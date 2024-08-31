import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import networkx as nx
import os
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np
import matplotlib.pyplot as plt

#GAT model
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.fc = torch.nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  #pooling layer
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

#load the graphs and labels
def load_graphs(graph_path, n_sub):
    graphs = []
    labels = []
    for sub_id in range(n_sub):
        for trial in range(80):  #80 trials per subject
            graph_file_path = os.path.join(graph_path, f'graph_sub_{sub_id}_trial_{trial}.graphml')
            if os.path.exists(graph_file_path):
                G = nx.read_graphml(graph_file_path)
                x = torch.tensor([[G.nodes[n]['feature']] for n in G.nodes], dtype=torch.float)
                edge_index = torch.tensor([[int(e[0]), int(e[1])] for e in G.edges], dtype=torch.long).t().contiguous()
                y = torch.tensor([int(G.graph['label'])], dtype=torch.long)
                
                #change labels for binary classification
                binary_label = 0 if y.item() in [0, 1] else 1
                
                data = Data(x=x, edge_index=edge_index, y=torch.tensor([binary_label], dtype=torch.long), batch=torch.zeros(x.size(0), dtype=torch.long))
                graphs.append(data)
                labels.append(binary_label)
            else:
                print(f"Graph file not found: {graph_file_path}")
    return graphs, labels

#training and evaluation
def train_and_evaluate(graphs, labels, n_sub):
    logo = LeaveOneGroupOut()
    groups = np.repeat(np.arange(n_sub), 80)  #80 trials per subject

    all_accuracies = []
    overall_correct = 0
    overall_total = 0

    for loso_number, (train_idx, test_idx) in enumerate(logo.split(graphs, labels, groups), start=1):
        train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=16, shuffle=True)
        test_loader = DataLoader([graphs[i] for i in test_idx], batch_size=16)

        model = GAT(input_dim=1, hidden_dim=16, output_dim=2, heads=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        #training
        model.train()
        for epoch in range(50):  #50 epochs
            epoch_loss = 0
            correct = 0
            total = 0
            for data in train_loader:
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y.view(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == data.y.view(-1)).sum().item()
                total += data.y.size(0)
            train_accuracy = correct / total * 100
            print(f"LOSO {loso_number}, Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        #evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y.view(-1)).sum().item()
                total += data.y.size(0)
        test_accuracy = correct / total * 100
        all_accuracies.append(test_accuracy)
        overall_correct += correct
        overall_total += total

        print(f"LOSO {loso_number}, Test Accuracy: {test_accuracy:.2f}%")

    overall_accuracy = overall_correct / overall_total * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return all_accuracies, overall_accuracy

#visualization
def plot_results(all_accuracies, overall_accuracy):
    plt.figure(figsize=(10, 6))
    plt.plot(all_accuracies, label='Test Accuracy per Subject')
    plt.axhline(y=overall_accuracy, color='r', linestyle='--', label='Overall Accuracy')
    plt.xlabel('Subject Index')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Subject and Overall Accuracy')
    plt.legend()
    plt.savefig('results/accuracy_plot_binary_gat.png')  #save the plot instead of showing it

#main execution
if __name__ == "__main__":
    graph_path = "graph_files"
    n_sub = 28
    graphs, labels = load_graphs(graph_path, n_sub)
    all_accuracies, overall_accuracy = train_and_evaluate(graphs, labels, n_sub)
    plot_results(all_accuracies, overall_accuracy)
