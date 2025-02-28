import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

# Load Cora dataset
dataset = Planetoid(root="data/Cora", name="Cora")

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define train function
def train(model, data, optimizer, epochs=200):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Define test function
def test(model, data):
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    pred = logits.argmax(dim=1)
    acc = (pred[data.test_mask] == data.y[data.test_mask].to(device)).sum().item() / data.test_mask.sum().item()
    print(f"Test Accuracy: {acc:.4f}")

# Train and evaluate each model
data = dataset[0].to(device)

# Train GCN
gcn = GCN(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
train(gcn, data, optimizer)
test(gcn, data)

# Train GAT
gat = GAT(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(gat.parameters(), lr=0.01, weight_decay=5e-4)
train(gat, data, optimizer)
test(gat, data)

# Train GraphSAGE
graphsage = GraphSAGE(dataset.num_node_features, 16, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(graphsage.parameters(), lr=0.01, weight_decay=5e-4)
train(graphsage, data, optimizer)
test(graphsage, data)
