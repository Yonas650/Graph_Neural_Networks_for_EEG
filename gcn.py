import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  #global pooling
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def load_graph_data(graph_path):
    data_list = []
    for file_name in os.listdir(graph_path):
        if file_name.endswith('.graphml'):
            file_path = os.path.join(graph_path, file_name)
            print(f"Loading file: {file_path}")
            G = nx.read_graphml(file_path)

            #node features
            features = []
            for node in G.nodes(data=True):
                features.append(float(node[1]['feature']))
            x = torch.tensor(features, dtype=torch.float).view(-1, 1) #shape (number_of_nodes, 1)
            print(f"Node features: {x}")

            #edges
            edges = [(int(u), int(v)) for u, v in G.edges()]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            print(f"Edge index: {edge_index}")

            #label
            try:
                label = int(file_name.split('_')[2].split('.')[0])  
            except ValueError as e:
                print(f"Error extracting label from {file_name}: {e}")
                continue
            y = torch.tensor([label], dtype=torch.long)
            print(f"Label: {y}")

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

    return data_list

def augment_data(data_list, augment_factor=10):
    augmented_data = []
    for data in data_list:
        augmented_data.append(data)
        for _ in range(augment_factor - 1):
            augmented_data.append(Data(
                x=data.x + torch.randn_like(data.x) * 0.01,
                edge_index=data.edge_index,
                y=data.y
            ))
    return augmented_data

def train(model, optimizer, criterion, loader):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def validate(model, criterion, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            total_loss += loss.item()
            pred = output.max(1)[1]
            correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

graph_path = "graph_files"
accuracy_results_path = "accuracy_results"
os.makedirs(accuracy_results_path, exist_ok=True)

data_list = load_graph_data(graph_path)
print(f"Loaded {len(data_list)} graphs.")

#data augmentation
data_list = augment_data(data_list)
print(f"Augmented data size: {len(data_list)}")

#kfold cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()

for fold, (train_index, test_index) in enumerate(kf.split(data_list)):
    print(f"FOLD {fold+1}")
    print("--------------------------------")
    train_data = [data_list[idx] for idx in train_index]
    test_data = [data_list[idx] for idx in test_index]

    #further split train_data into train and val
    val_split = int(len(train_data) * 0.2)
    val_data = train_data[:val_split]
    train_data = train_data[val_split:]

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    model = GCN(input_dim=1, hidden_dim=16, output_dim=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []
    test_accuracies = []

    for epoch in range(1, 201):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        val_loss, val_acc = validate(model, criterion, val_loader)

        if epoch % 10 == 0:
            test_loss, test_acc = validate(model, criterion, test_loader)
            test_accuracies.append(test_acc)
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    results.append(test_accuracies)

    #save loss and accuracy graphs
    epochs = range(1, 201)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold+1} Loss')
    plt.legend()
    plt.savefig(os.path.join(accuracy_results_path, f'loss_fold_{fold+1}.png'))
    plt.close()

#average performance over folds
avg_test_accuracies = np.mean(results, axis=0)
plt.figure(figsize=(10, 5))
plt.plot(range(10, 201, 10), avg_test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Test Accuracy over Epochs')
plt.legend()
plt.savefig(os.path.join(accuracy_results_path, 'average_test_accuracy.png'))
plt.show()

print("K-Fold Cross-Validation results:")
overall_accuracy = 0
for fold, accuracies in enumerate(results):
    fold_avg_acc = np.mean(accuracies)
    print(f"FOLD {fold+1} - Average Test Accuracy: {fold_avg_acc:.4f}")
    overall_accuracy += fold_avg_acc
overall_accuracy /= len(results)
print(f"Overall Model Accuracy: {overall_accuracy:.4f}")
